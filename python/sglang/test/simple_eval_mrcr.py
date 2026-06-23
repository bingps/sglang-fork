"""
MRCR v2: Multi-Round Coreference Resolution
OpenAI long-context benchmark testing multi-needle retrieval across conversations.
https://huggingface.co/datasets/openai/mrcr
"""

import difflib
import json
from typing import Any, Dict, List, Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

DEFAULT_DATASET = "openai/mrcr"
DEFAULT_SPLIT = "train"


class MRCREval(Eval):
    def __init__(
        self,
        num_examples: Optional[int] = None,
        num_threads: int = 1,
        n_needles: Optional[int] = None,
        min_context_length: Optional[int] = None,
        max_context_length: Optional[int] = None,
    ):
        examples = self._load_hf_dataset()

        if n_needles is not None:
            examples = [ex for ex in examples if ex.get("n_needles") == n_needles]

        if min_context_length is not None:
            examples = [
                ex for ex in examples if ex.get("n_chars", 0) >= min_context_length
            ]

        if max_context_length is not None:
            examples = [
                ex for ex in examples if ex.get("n_chars", 0) <= max_context_length
            ]

        if num_examples:
            examples = examples[: min(num_examples, len(examples))]

        if not examples:
            raise ValueError("No examples available for MRCR evaluation after filtering")

        self.examples = examples
        self.num_threads = num_threads

        print(f"Loaded {len(self.examples)} examples from MRCR")
        if n_needles is not None:
            print(f"Filtered to n_needles={n_needles}")

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
            prompt_str = row["prompt"]
            try:
                messages = json.loads(prompt_str)
            except (json.JSONDecodeError, TypeError):
                messages = [
                    sampler._pack_message(content=prompt_str, role="user")
                ]

            if not isinstance(messages, list):
                messages = [
                    sampler._pack_message(content=str(messages), role="user")
                ]

            prompt_messages = [
                sampler._pack_message(content=m["content"], role=m["role"])
                for m in messages
            ]

            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""

            answer = row.get("answer", "")
            prefix = row.get("random_string_to_prepend", "")

            if prefix and not response_text.startswith(prefix):
                score = 0.0
            else:
                resp_stripped = response_text[len(prefix) :] if prefix else response_text
                ans_stripped = answer[len(prefix) :] if prefix and answer.startswith(prefix) else answer
                score = difflib.SequenceMatcher(
                    None, resp_stripped, ans_stripped
                ).ratio()

            n_needles = row.get("n_needles", 0)
            n_chars = row.get("n_chars", 0)

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages[-2:],
                next_message=dict(content=response_text[:500], role="assistant"),
                score=score,
                correct_answer=answer[:200],
                extracted_answer=response_text[:200],
            )

            metrics = {
                "chars": len(response_text),
                f"needles_{n_needles}": score,
            }

            if n_chars < 32000:
                metrics["ctx_short"] = score
            elif n_chars < 131000:
                metrics["ctx_medium"] = score
            else:
                metrics["ctx_long"] = score

            return SingleEvalResult(
                html=html,
                score=score,
                convo=prompt_messages[-2:]
                + [dict(content=response_text[:500], role="assistant")],
                metrics=metrics,
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
