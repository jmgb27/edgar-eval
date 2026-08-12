"""The eval script must not fail on a cosmetic line.

Regression: `run_eval.py` printed every metric correctly and then exited 1,
because its closing "wrote results to X" line called Path.relative_to with a
path that was not under the repository root. Locally the ablation runs piped
output to /dev/null and never checked the exit status, so the crash stayed
invisible until CI -- where the exit code is the entire point, and it skipped the
threshold check that is the gate's reason to exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edgar_eval.config import REPO_ROOT


def _display_path(out: Path) -> Path:
    """Mirror of the logic in scripts/run_eval.py."""
    try:
        return out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return out


@pytest.mark.parametrize(
    "raw",
    [
        "eval/results/pr.json",  # relative — what CI passes
        "/tmp/elsewhere.json",  # absolute, outside the repo
        "../outside.json",  # above the repo root
    ],
)
def test_display_path_never_raises(raw: str) -> None:
    assert isinstance(_display_path(Path(raw)), Path)


def test_paths_inside_the_repo_are_shortened() -> None:
    """The nicety still works where it can; it just cannot be load-bearing."""
    assert _display_path(REPO_ROOT / "eval" / "results" / "x.json") == Path("eval/results/x.json")
