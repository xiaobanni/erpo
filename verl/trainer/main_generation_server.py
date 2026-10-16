# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm as tqdm_async

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


# Cap client-side concurrency (on the order of vLLM's max_num_seqs) to avoid
# ephemeral-port exhaustion (EADDRNOTAVAIL) when the request fanout is large.
_SHARED_CONNECTION_LIMIT = 1024


async def submit_request(server_address, *, session=None, **chat_complete_request):
    """POST a single chat-completion request.

    If ``session`` is provided, reuse it (preferred path for large-fanout
    callers; see ``generate_per_replica``). If ``session`` is None, fall back to
    a per-call ClientSession for single-shot callers.
    """
    extra_headers = chat_complete_request.pop("extra_headers", {})
    headers = {"Authorization": "Bearer token-abc123", **extra_headers}
    url = f"http://{server_address}/v1/chat/completions"

    if session is not None:
        async with session.post(url=url, headers=headers, json=chat_complete_request) as resp:
            data = await resp.json()
            return ChatCompletion(**data)

    timeout = aiohttp.ClientTimeout(total=None)
    own_session = aiohttp.ClientSession(timeout=timeout)
    try:
        async with own_session.post(url=url, headers=headers, json=chat_complete_request) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await own_session.close()


async def generate_per_replica(
    server_address,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
    chat_template_kwargs: dict | None = None,
):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    request_template = dict(sampling_params)
    if chat_template_kwargs:
        request_template["chat_template_kwargs"] = chat_template_kwargs

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **request_template,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    connector = aiohttp.TCPConnector(limit=_SHARED_CONNECTION_LIMIT)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [submit_request(server_address, session=session, **req) for req in chat_complete_request]
        results = await tqdm_async.gather(
            *tasks,
            desc=f"gen@{server_address}",
            unit="req",
        )
    return results


async def generate(
    server_addresses: list,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_numpy: np.ndarray,
    chat_template_kwargs: dict | None = None,
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    results = await asyncio.gather(
        *[
            generate_per_replica(
                server_addresses[i],
                model_path,
                n_samples,
                sampling_params,
                chat_sub_array[i],
                chat_template_kwargs=chat_template_kwargs,
            )
            for i in range(num_replicas)
        ]
    )
    return results


def extract_completion_tokens(result) -> int:
    usage = getattr(result, "usage", None)
    if usage is None:
        raise RuntimeError("vLLM response is missing usage metadata; cannot derive completion token length")

    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    else:
        completion_tokens = getattr(usage, "completion_tokens", None)

    if completion_tokens is None:
        raise RuntimeError("vLLM response usage is missing completion_tokens; cannot derive completion token length")

    return int(completion_tokens)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}})

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        "top_k": config.actor_rollout_ref.rollout.top_k,
        "min_p": config.actor_rollout_ref.rollout.min_p,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
    }
    chat_template_kwargs = OmegaConf.to_container(config.data.get("apply_chat_template_kwargs", {}), resolve=True)
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    if not isinstance(chat_template_kwargs, dict):
        raise TypeError(f"data.apply_chat_template_kwargs must resolve to a dict, got {type(chat_template_kwargs)}")

    from omegaconf import ListConfig

    train_files = config.data.train_files
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

    datasets = []
    for train_file in train_files:
        dataset = pd.read_parquet(train_file)
        datasets.append(dataset)

    # concat dataset
    dataset = pd.concat(datasets, axis=0, ignore_index=True)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    chat_numpy = np.array(chat_lst)

    # start native server
    server_handles, server_addresses = asyncio.run(start_server(config))

    # run generate
    gen_results = asyncio.run(
        generate(
            server_addresses,
            config.actor_rollout_ref.model.path,
            n_samples,
            sampling_params,
            chat_numpy,
            chat_template_kwargs=chat_template_kwargs,
        )
    )

    # reshape results into a numpy array
    import itertools

    flat_results = list(itertools.chain.from_iterable(gen_results))

    # extract content from results
    results = np.array([result.choices[0].message.content for result in flat_results])
    completion_token_lengths = np.array([extract_completion_tokens(result) for result in flat_results])
    results = np.reshape(results, (-1, n_samples))
    completion_token_lengths = np.reshape(completion_token_lengths, (-1, n_samples))

    assert results.shape == (len(chat_lst), n_samples)
    assert completion_token_lengths.shape == (len(chat_lst), n_samples)

    results = results.tolist()
    completion_token_lengths = completion_token_lengths.tolist()

    # Render the prompt with the same chat_template_kwargs passed to vLLM so the
    # saved `rendered_prompt` matches what the model saw at generation time.
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        config.actor_rollout_ref.model.path, trust_remote_code=True
    )
    rendered_prompts = [
        _tokenizer.apply_chat_template(
            list(chat), tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        for chat in chat_lst
    ]

    # add to the data frame
    dataset["responses"] = results
    dataset["response_token_lengths"] = completion_token_lengths
    dataset["rendered_prompt"] = rendered_prompts

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
