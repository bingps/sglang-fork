"""
GraphWalks: Graph traversal benchmark for long-context evaluation.
Models receive a directed graph edge list and perform BFS or find parent nodes.
https://huggingface.co/datasets/openai/graphwalks
"""

import re
from typing import Any, Dict, List, Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

DEFAULT_DATASET = "openai/graphwalks"
DEFAULT_SPLIT = "train"


def extract_final_answer(response: str) -> Optional[List[str]]:
    for line in reversed(response.strip().splitlines()):
        match = re.search(r"Final Answer:\s*\[([^\]]*)\]", line, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            if not raw:
                return []
            return [n.strip().strip("'\"") for n in raw.split(",") if n.strip()]
    return None


def compute_f1(predicted: List[str], expected: List[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    pred_set = set(predicted)
    exp_set = set(expected)
    overlap = len(pred_set & exp_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(exp_set)
    return 2 * precision * recall / (precision + recall)


class GraphWalksEval(Eval):
    def __init__(
        self,
        num_examples: Optional[int] = None,
        num_threads: int = 1,
        problem_type: Optional[str] = None,
        max_prompt_chars: Optional[int] = None,
    ):
        examples = self._load_hf_dataset()

        if problem_type is not None:
            examples = [
                ex for ex in examples if ex.get("problem_type") == problem_type
            ]

        if max_prompt_chars is not None:
            examples = [
                ex
                for ex in examples
                if ex.get("prompt_chars", len(ex.get("prompt", ""))) <= max_prompt_chars
            ]

        if num_examples:
            examples = examples[: min(num_examples, len(examples))]

        if not examples:
            raise ValueError(
                "No examples available for GraphWalks evaluation after filtering"
            )

        self.examples = examples
        self.num_threads = num_threads

        print(f"Loaded {len(self.examples)} examples from GraphWalks")
        if problem_type:
            print(f"Filtered to problem_type={problem_type}")

    def _load_hf_dataset(self) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Please install the 'datasets' package: pip install datasets"
            ) from exc

        dataset = load_dataset(DEFAULT_DATASET, split=DEFAULT_SPLIT)
        return [dict(row) for row in dataset]

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt = row["prompt"]
            prompt_messages = [
                sampler._pack_message(content=prompt, role="user")
            ]

            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""

            answer_nodes = row.get("answer_nodes", [])
            predicted = extract_final_answer(response_text)

            if predicted is None:
                score = 0.0
                predicted = []
            else:
                score = compute_f1(predicted, answer_nodes)

            problem_type = row.get("problem_type", "unknown")
            prompt_chars = row.get("prompt_chars", len(prompt))

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=[
                    sampler._pack_message(
                        content=f"[GraphWalks {problem_type} | {prompt_chars} chars]",
                        role="user",
                    )
                ],
                next_message=dict(content=response_text[-500:], role="assistant"),
                score=score,
                correct_answer=str(answer_nodes[:20]),
                extracted_answer=str(predicted[:20]),
            )

            metrics = {
                "chars": len(response_text),
                f"type_{problem_type}": score,
            }

            return SingleEvalResult(
                html=html,
                score=score,
                convo=[
                    sampler._pack_message(
                        content=f"[GraphWalks {problem_type}]", role="user"
                    ),
                    dict(content=response_text[-500:], role="assistant"),
                ],
                metrics=metrics,
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
