"""
RULER: What's the Real Context Size of Your Long-Context Language Models?
Synthetic benchmark with retrieval, multi-hop tracing, aggregation, and QA tasks.
https://huggingface.co/datasets/rbiswasfc/ruler
https://arxiv.org/abs/2404.06654
"""

from typing import Any, Dict, List, Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

DEFAULT_DATASET = "rbiswasfc/ruler"
DEFAULT_SPLIT = "validation"

ALL_CONFIGS = [
    "niah_multikey_1_4k",
    "niah_multikey_1_8k",
    "qa_2_4k",
    "qa_2_8k",
    "vt_4k",
    "vt_8k",
    "cwe_4k",
    "cwe_8k",
]

TASK_NAMES = {
    "niah_multikey_1": "Multi-key NIAH",
    "qa_2": "QA (HotpotQA)",
    "vt": "Variable Tracking",
    "cwe": "Common Words",
}


def score_ruler_response(response: str, outputs: List[str]) -> float:
    response_lower = response.lower()
    for expected in outputs:
        if expected and expected.lower() in response_lower:
            return 1.0
    return 0.0


class RULEREval(Eval):
    def __init__(
        self,
        num_examples: Optional[int] = None,
        num_threads: int = 1,
        ruler_tasks: Optional[str] = None,
        ruler_length: Optional[str] = None,
    ):
        configs = self._resolve_configs(ruler_tasks, ruler_length)
        examples = self._load_hf_dataset(configs)

        if num_examples:
            examples = examples[: min(num_examples, len(examples))]

        if not examples:
            raise ValueError(
                "No examples available for RULER evaluation after filtering"
            )

        self.examples = examples
        self.num_threads = num_threads

        print(f"Loaded {len(self.examples)} examples from RULER")
        print(f"Configs: {configs}")

    def _resolve_configs(
        self, ruler_tasks: Optional[str], ruler_length: Optional[str]
    ) -> List[str]:
        if ruler_tasks:
            configs = [t.strip() for t in ruler_tasks.split(",")]
            invalid = [c for c in configs if c not in ALL_CONFIGS]
            if invalid:
                raise ValueError(
                    f"Invalid RULER configs: {invalid}. Valid: {ALL_CONFIGS}"
                )
            return configs

        if ruler_length:
            return [c for c in ALL_CONFIGS if c.endswith(f"_{ruler_length}")]

        return ALL_CONFIGS

    def _load_hf_dataset(self, configs: List[str]) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Please install the 'datasets' package: pip install datasets"
            ) from exc

        all_examples = []
        for config in configs:
            dataset = load_dataset(DEFAULT_DATASET, config, split=DEFAULT_SPLIT)
            for row in dataset:
                example = dict(row)
                example["_config"] = config
                task_key = "_".join(config.split("_")[:-1])
                example["_task"] = TASK_NAMES.get(task_key, task_key)
                all_examples.append(example)

        return all_examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt = row["input"]
            prompt_messages = [
                sampler._pack_message(content=prompt, role="user")
            ]

            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""

            outputs = row.get("outputs", [])
            score = score_ruler_response(response_text, outputs)

            config = row.get("_config", "unknown")
            task = row.get("_task", "unknown")

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=[
                    sampler._pack_message(
                        content=f"[RULER {config}] {prompt[:200]}...",
                        role="user",
                    )
                ],
                next_message=dict(content=response_text[:500], role="assistant"),
                score=score,
                correct_answer=str(outputs[:3]),
                extracted_answer=response_text[:200],
            )

            metrics = {
                "chars": len(response_text),
                f"task_{config}": score,
            }

            return SingleEvalResult(
                html=html,
                score=score,
                convo=[
                    sampler._pack_message(
                        content=f"[RULER {config}]", role="user"
                    ),
                    dict(content=response_text[:500], role="assistant"),
                ],
                metrics=metrics,
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
