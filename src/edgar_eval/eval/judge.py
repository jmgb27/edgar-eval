"""The faithfulness judge, and the honesty about what it is.

An LLM judge is a measuring instrument, and an uncalibrated instrument produces
numbers that look like evidence without being evidence. Every faithfulness score
this repo publishes comes from this judge, so the judge itself is measured
against human labels before it is allowed to gate anything --
`scripts/calibrate_judge.py` reports its TPR and TNR and the report is committed.

The judge is Claude Haiku 4.5 while generation is Qwen, deliberately. A judge
drawn from the same family as the model it grades shares that model's blind
spots, and the resulting score flatters itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from edgar_eval.llm import build_judge

# Versioned and hashed into every result. A judge prompt edit moves every
# faithfulness number, and a move nobody can attribute is indistinguishable
# from a regression.
FAITHFULNESS_SYSTEM = """You check whether an answer is supported by the passages given to it.

An answer is SUPPORTED only if every factual claim in it -- every figure, name,
date and comparison -- is stated in or directly derivable from the passages.

Judge only against the passages. Do not use anything you know about the company.
A passage on the same topic is not support for a specific number: if the answer
says a figure the passages do not contain, the answer is UNSUPPORTED even when
the figure is plausible or well known.

Return JSON only:
  {"verdict": "supported" | "unsupported",
   "unsupported_claims": [claims not supported by any passage],
   "reason": "one sentence"}"""


def judge_prompt_sha256() -> str:
    return hashlib.sha256(FAITHFULNESS_SYSTEM.encode("utf-8")).hexdigest()[:16]


@dataclass
class Verdict:
    supported: bool
    unsupported_claims: list[str]
    reason: str
    raw: str = ""

    @property
    def label(self) -> str:
        return "GROUNDED" if self.supported else "UNGROUNDED"


def _parse(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def judge_faithfulness(
    *, question: str, answer: str, contexts: list[str], llm: Any | None = None
) -> Verdict:
    """Is `answer` supported by `contexts`?

    A parse failure returns *supported* rather than raising, and that default is
    chosen deliberately: an unparseable judgement is a broken instrument, and a
    broken instrument must not manufacture a hallucination finding. The
    calibration run counts these, so a judge that starts failing to parse shows
    up as degraded TNR rather than as a sudden crop of false alarms.
    """
    llm = llm or build_judge(max_tokens=800)
    passages = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    response = llm.invoke(
        [
            ("system", FAITHFULNESS_SYSTEM),
            ("human", f"Question: {question}\n\nAnswer:\n{answer}\n\nPassages:\n{passages}"),
        ]
    )
    raw = str(response.content)
    parsed = _parse(raw)

    verdict = str(parsed.get("verdict", "supported")).lower()
    return Verdict(
        supported=verdict != "unsupported",
        unsupported_claims=[
            c for c in (parsed.get("unsupported_claims") or []) if isinstance(c, str)
        ],
        reason=str(parsed.get("reason", "")),
        raw=raw,
    )


@dataclass
class ConfusionMatrix:
    """Positive class = UNGROUNDED, i.e. the judge catching a hallucination.

    Framed that way because the question a reader actually has is "how often
    does this judge catch a made-up number, and how often does it cry wolf" --
    not "how often does it agree".
    """

    tp: int = 0  # ungrounded, judged ungrounded
    fp: int = 0  # grounded, judged ungrounded   (false alarm)
    fn: int = 0  # ungrounded, judged grounded   (missed hallucination)
    tn: int = 0  # grounded, judged grounded

    def add(self, *, truth_ungrounded: bool, judged_ungrounded: bool) -> None:
        if truth_ungrounded and judged_ungrounded:
            self.tp += 1
        elif truth_ungrounded:
            self.fn += 1
        elif judged_ungrounded:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def _ratio(self, numerator: int, denominator: int) -> float | None:
        # None rather than 0.0: a metric with an empty denominator is unknown,
        # and reporting it as zero would make an unmeasured judge look terrible
        # instead of unmeasured.
        return None if denominator == 0 else round(numerator / denominator, 4)

    @property
    def tpr(self) -> float | None:
        """Recall on hallucinations: of the made-up answers, how many were caught."""
        return self._ratio(self.tp, self.tp + self.fn)

    @property
    def tnr(self) -> float | None:
        """Of the sound answers, how many were left alone."""
        return self._ratio(self.tn, self.tn + self.fp)

    @property
    def precision(self) -> float | None:
        return self._ratio(self.tp, self.tp + self.fp)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.tpr
        if p is None or r is None or (p + r) == 0:
            return None
        return round(2 * p * r / (p + r), 4)

    @property
    def accuracy(self) -> float | None:
        return self._ratio(self.tp + self.tn, self.n)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "tpr": self.tpr,
            "tnr": self.tnr,
            "precision": self.precision,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }
