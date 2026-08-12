"""Build the judge-calibration set from the curated gold set.

Calibrating a judge needs (question, answer, contexts) triples whose true label
is *known*. Getting that from generated answers means a human deciding, case by
case, whether each one is faithful -- slow, and only as reliable as the human's
attention on the fiftieth example.

Instead each triple is built from material already verified to exist in the
corpus, in matched pairs:

  GROUNDED    the gold reference answer, with the passage that supports it
  UNGROUNDED  the same answer with exactly one fact corrupted, same passage

The label is then certain by construction, not by judgement, and the pair is
matched -- identical question, identical passage, one altered figure -- so the
judge is tested on the thing that actually matters: noticing that a number
changed. Every corruption is recorded in the output so a reviewer can confirm
the label at a glance rather than re-deriving it.

Balanced 50/50 on purpose: an unbalanced set lets a judge that answers
"supported" every time post a flattering accuracy.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from edgar_eval.config import REPO_ROOT
from edgar_eval.db import close_pool, connection
from edgar_eval.eval.retrieval_metrics import chunk_matches_span

GOLD = REPO_ROOT / "eval" / "golden" / "curated.jsonl"
OUT = REPO_ROOT / "eval" / "labels" / "judge_calibration.jsonl"

_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
# A bare four-digit year is not a usable corruption target. Turning 2022 into
# 9022 reads as a typo rather than a fabricated fact, and a judge that says so
# is being reasonable -- so the label would be wrong, not the judge. Money,
# percentages and any figure carrying a decimal or thousands separator are
# corrupted instead.
_BARE_YEAR = re.compile(r"^(?:19|20)\d{2}$")
PER_SPAN_CONTEXTS = 2

# For answers carrying no figure to corrupt, a claim that contradicts the
# passage outright. Keyed by gold question id so each is deliberate rather than
# generated, and so a reviewer can audit the list directly.
_CONTRADICTIONS: dict[str, str] = {
    "sf-04": "Apple reported three unresolved staff comments from the SEC.",
    "sf-05": "Apple disclosed two mine safety violations during fiscal 2023.",
    "sf-06": "Microsoft disclosed one mine safety violation during fiscal 2023.",
    "sf-07": "Apple's reportable segments are iPhone, Mac, iPad and Services.",
    "sf-08": "Microsoft reported four unresolved written comments from the SEC staff.",
    "cmp-01": "Apple reports the item as not applicable; Microsoft discloses two violations.",
    "cmp-02": "Both companies report along identical geographic segments.",
    "cmp-03": "Neither company identifies international operations as a risk.",
    "cmp-04": "Neither company discusses taxation of foreign earnings as a risk.",
    "rsk-01": "Apple states that every component it uses is available from at least five suppliers.",
    "rsk-02": "Microsoft states that it has no artificial intelligence investments.",
    "rsk-03": "Microsoft states that employee retention presents no risk to its business.",
}


def _corrupt_number(text: str) -> tuple[str, str] | None:
    """Alter the first figure in `text`, returning (corrupted, description)."""
    match = next(
        (m for m in _NUMBER.finditer(text) if not _BARE_YEAR.match(m.group(0))),
        None,
    )
    if not match:
        return None
    original = match.group(0)
    digits = re.sub(r"[^\d]", "", original)
    if not digits:
        return None

    # Change the leading digit so the figure is unmistakably different rather
    # than a rounding difference the judge could reasonably tolerate.
    leading = digits[0]
    replacement_digit = "9" if leading != "9" else "4"
    corrupted_digits = replacement_digit + digits[1:]

    corrupted = original
    for a, b in zip(digits, corrupted_digits, strict=True):
        if a != b:
            corrupted = original.replace(a, b, 1)
            break

    return (
        text[: match.start()] + corrupted + text[match.end() :],
        f"{original} -> {corrupted}",
    )


def main() -> int:
    questions = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]

    with connection() as conn:
        rows = conn.execute("SELECT id, text FROM chunks").fetchall()
    chunks = {r["id"]: r["text"] for r in rows}
    close_pool()

    triples: list[dict[str, Any]] = []
    skipped: list[str] = []

    for q in questions:
        if q["category"] == "unanswerable":
            continue
        spans = q.get("reference_contexts") or []
        # Collect per span, not across all of them. A globally-truncated list
        # lets one promiscuous span crowd out the evidence for the others: the
        # phrase "Productivity and Business Processes" matches several Microsoft
        # chunks, so a flat [:3] silently dropped the Apple chunk from a
        # comparison question and left it labelled GROUNDED with evidence that
        # did not support it. The judge caught exactly that, correctly.
        contexts: list[str] = []
        for span in spans:
            matches = [text for text in chunks.values() if chunk_matches_span(text, span)]
            contexts.extend(matches[:PER_SPAN_CONTEXTS])
        contexts = list(dict.fromkeys(contexts))
        if not contexts:
            skipped.append(f"{q['id']}: no supporting chunk found")
            continue

        corruption = _corrupt_number(q["reference"])
        if corruption is None:
            contradiction = _CONTRADICTIONS.get(q["id"])
            if contradiction is None:
                skipped.append(f"{q['id']}: no figure to corrupt and no contradiction defined")
                continue
            corrupted_answer, description = contradiction, "contradicts the passage"
        else:
            corrupted_answer, description = corruption

        triples.append(
            {
                "id": f"{q['id']}-grounded",
                "source_question_id": q["id"],
                "question": q["question"],
                "answer": q["reference"],
                "contexts": contexts,
                "label": "GROUNDED",
                "why": "verbatim gold reference answer, with its supporting passage",
            }
        )
        triples.append(
            {
                "id": f"{q['id']}-ungrounded",
                "source_question_id": q["id"],
                "question": q["question"],
                "answer": corrupted_answer,
                "contexts": contexts,
                "label": "UNGROUNDED",
                "why": f"gold answer with one fact corrupted: {description}",
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(t) + "\n" for t in triples))

    grounded = sum(1 for t in triples if t["label"] == "GROUNDED")
    print(f"\n  {len(triples)} triples -> {OUT.relative_to(REPO_ROOT)}")
    print(f"    GROUNDED   {grounded}")
    print(f"    UNGROUNDED {len(triples) - grounded}")
    for s in skipped:
        print(f"    skipped: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
