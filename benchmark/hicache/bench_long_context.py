import json
import queue
import time

import requests
from bench_multiturn import (
    ReadyQueue,
    WorkloadGenerator,
    gen_payload,
    log_to_jsonl_file,
    parse_args,
)
from tqdm.asyncio import tqdm

from sglang.bench_serving import get_tokenizer


class ContextWorkloadGenerator(WorkloadGenerator):
    def __init__(self, args):
        # Construct the base URL for requests
        self.baseurl = f"http://{args.host}:{args.port}/"
        self.url = self.baseurl + "generate"

        self.dataset_format = args.dataset_format

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.sent_requests = 0
        self.completed_requests = 0

        if self.dataset_format == "loopserve":
            init_requests = self._build_loopserve_requests(args)
            queue_policy = "fifo"
        else:
            init_requests = self._build_requests_from_json(args)
            queue_policy = args.ready_queue_policy

        if not init_requests:
            raise ValueError("No requests prepared for benchmarking")

        self.ready_queue = ReadyQueue(init_requests=init_requests, policy=queue_policy)

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=len(init_requests))
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "itl": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }

        self.max_parallel = args.max_parallel
        self.logfile = args.log_file
        self.enable_round_barrier = False

    def _build_requests_from_json(self, args):
        with open(args.dataset_path) as f:
            dataset = json.load(f)

        num_requests = min(args.num_clients, len(dataset["queries"]))
        init_requests = []
        for i in range(num_requests):
            context_id = dataset["queries"][i]["context"]
            prompt = dataset["contexts"][context_id] + dataset["queries"][i]["question"]
            answer_len = len(
                self.tokenizer(dataset["queries"][i]["reference_answer"])["input_ids"]
            )
            init_requests.append((i, gen_payload(prompt, answer_len, args.lora_path)))
        return init_requests

    def _build_loopserve_requests(self, args):
        from datasets import load_dataset

        dataset = load_dataset(
            "TreeAILab/Multi-turn_Long-context_Benchmark_for_LLMs",
            "multi-turn_QA",
            split="train",
        )

        num_conversations = min(args.num_clients, len(dataset))
        init_requests = []
        request_id = 0
        for _ in range(args.num_rounds):
            for idx in range(num_conversations):
                all_prompt = ""
                for ti, turn in enumerate(dataset[idx]["multi_turn"]):
                    prompt = turn["prompt_back"]
                    question = turn["question_at_back"]
                    assert question in prompt
                    prompt_no_question = prompt.replace(question, "", 1)
                    assert prompt_no_question == prompt[: -len(question)]

                    init_requests.append(
                        (
                            request_id,
                            gen_payload(
                                all_prompt + prompt_no_question,
                                args.output_length,
                                args.lora_path,
                            ),
                        )
                    )
                    request_id += 1

                    init_requests.append(
                        (
                            request_id,
                            gen_payload(
                                all_prompt + prompt,
                                args.output_length,
                                args.lora_path,
                            ),
                        )
                    )
                    request_id += 1
                    all_prompt += prompt

        return init_requests

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(response.itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.performance_metrics["generated_len"].append(response.generated_len)
                self.completed_requests += 1

            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break


if __name__ == "__main__":
    args = parse_args()
    args.max_parallel = 24
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    if args.disable_auto_run:
        print("Running with specified request rate...")
        request_rates = [args.request_rate]
    else:
        print("Auto-running with different request rates...")
        request_rates = [24, 16, 12, 8, 4, 2, 1]

    for request_rate in request_rates:
        args.request_rate = request_rate
        requests.post(flush_cache_url)
        time.sleep(1)
        performance_data = ContextWorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, args.tag)
