"""Item sectionising.

These are the highest-value tests in the repo. If sectionising is wrong,
every downstream number is wrong in a way no eval metric would attribute
correctly: filtered retrieval quietly returns the wrong section, and the
benchmark still produces a plausible-looking table.
"""

from __future__ import annotations

import pytest

from edgar_eval.ingest.sections import (
    ITEM_TITLES_10K,
    ITEM_TITLES_10Q,
    ItemRef,
    normalise_item_number,
    parse_item_heading,
    parse_part_heading,
    sectionise,
)


# ── heading parsing ─────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Item 1. Business", "1"),
        ("ITEM 1A. RISK FACTORS", "1A"),
        ("Item 1A - Risk Factors", "1A"),
        ("Item 1A – Risk Factors", "1A"),  # en dash
        ("Item 1A — Risk Factors", "1A"),  # em dash
        ("Item 7A: Quantitative and Qualitative Disclosures About Market Risk", "7A"),
        ("ITEM 15 EXHIBITS, FINANCIAL STATEMENT SCHEDULES", "15"),
        ("Item 6 [Reserved]", "6"),
        ("Item 9B.", "9B"),
        ("  item   7   ", "7"),  # sloppy whitespace
        ("Item 1a. Risk Factors", "1A"),  # lowercase suffix normalises
        ("PART II - ITEM 7. MD&A", "7"),  # Part inlined on the same line
    ],
)
def test_parses_real_heading_shapes(text: str, expected: str) -> None:
    parsed = parse_item_heading(text)
    assert parsed is not None, f"failed to parse: {text!r}"
    assert parsed[0] == expected


@pytest.mark.parametrize(
    "text",
    [
        "as described in Item 1A above, our business is subject to risks",
        "See Item 7 for a discussion of these trends.",
        "The itemized list follows.",
        "",
        "   ",
        # A whole paragraph that merely opens with the word: the length guard
        # and the anchored title group both have to hold here.
        "Item 1A risk factors are discussed at length throughout this report, "
        "including in our quarterly filings and in the accompanying notes to "
        "the consolidated financial statements, which describe in detail the "
        "matters that management believes are most significant to investors.",
    ],
)
def test_rejects_prose_that_merely_mentions_an_item(text: str) -> None:
    assert parse_item_heading(text) is None


def test_normalise_item_number() -> None:
    assert normalise_item_number("7", None) == "7"
    assert normalise_item_number("7", "a") == "7A"
    assert normalise_item_number("1", "B") == "1B"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("PART I", "I"), ("Part II", "II"), ("PART III.", "III"), ("Part IV", "IV")],
)
def test_parses_part_heading(text: str, expected: str) -> None:
    assert parse_part_heading(text) == expected


def test_part_heading_does_not_match_prose() -> None:
    assert parse_part_heading("Part of our strategy is to grow") is None


# ── table of contents suppression ───────────────────────────
def _toc_then_body() -> tuple[list[str], list[str]]:
    """A filing shaped like a real 10-K: TOC first, then the body."""
    toc = [
        "Table of Contents",
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 2. Properties",
        "Item 3. Legal Proceedings",
        "Item 7. Management's Discussion and Analysis",
        "Item 8. Financial Statements",
    ]
    body: list[str] = []
    for num, title in [
        ("1", "Business"),
        ("1A", "Risk Factors"),
        ("2", "Properties"),
        ("3", "Legal Proceedings"),
        ("7", "Management's Discussion and Analysis"),
        ("8", "Financial Statements"),
    ]:
        body.append(f"Item {num}. {title}")
        body.extend(f"Body paragraph {i} of item {num}." for i in range(12))

    texts = toc + body
    categories = ["Title"] * len(toc)
    for _ in range(len(body)):
        categories.append("NarrativeText")
    # Mark the real body headings as Titles.
    for i, t in enumerate(texts):
        if i >= len(toc) and parse_item_heading(t):
            categories[i] = "Title"
    return texts, categories


def test_table_of_contents_is_skipped() -> None:
    """The classic failure: every Item is "found" in the TOC, so the entire
    body is assigned to whichever Item the TOC listed last."""
    texts, categories = _toc_then_body()
    result = sectionise(texts, form_type="10-K", categories=categories)

    assert result.toc_end_index > 0, "TOC was not detected"
    # Nothing inside the TOC is assigned to an Item.
    assert all(item is None for item in result.items[: result.toc_end_index])
    # Each body heading is a real heading, counted once.
    assert [h.item.number for h in result.headings] == ["1", "1A", "2", "3", "7", "8"]


def test_body_content_lands_in_the_right_item() -> None:
    texts, categories = _toc_then_body()
    result = sectionise(texts, form_type="10-K", categories=categories)

    for i, text in enumerate(texts):
        if text.startswith("Body paragraph") and "of item 1A" in text:
            assert result.items[i] is not None
            assert result.items[i].number == "1A"  # type: ignore[union-attr]


def test_no_toc_detected_when_document_has_no_toc() -> None:
    """A short excerpt with one Item heading must not be mistaken for a TOC."""
    texts = ["Item 7. Management's Discussion and Analysis"] + [f"Paragraph {i}" for i in range(30)]
    categories = ["Title"] + ["NarrativeText"] * 30
    result = sectionise(texts, form_type="10-K", categories=categories)

    assert result.toc_end_index == 0
    assert len(result.headings) == 1
    assert result.items[5] is not None
    assert result.items[5].number == "7"  # type: ignore[union-attr]


# ── Part disambiguation ─────────────────────────────────────
def test_10q_item_1_is_disambiguated_by_part() -> None:
    """Part I Item 1 is Financial Statements; Part II Item 1 is Legal
    Proceedings. Keying on the number alone silently merges them."""
    texts = [
        "PART I",
        "Item 1. Financial Statements",
        "Revenue for the quarter was $1,000.",
        "PART II",
        "Item 1. Legal Proceedings",
        "We are party to various claims.",
    ]
    categories = ["Title", "Title", "NarrativeText", "Title", "Title", "NarrativeText"]
    result = sectionise(texts, form_type="10-Q", categories=categories)

    financial = result.items[2]
    legal = result.items[5]
    assert financial is not None and legal is not None
    assert financial.key == "I-1"
    assert legal.key == "II-1"
    assert financial != legal, "the two Item 1s must not collapse into one section"
    assert financial.title("10-Q") == "Financial Statements"
    assert legal.title("10-Q") == "Legal Proceedings"


def test_10k_part_is_inferred_when_no_part_heading_is_present() -> None:
    """Many 10-Ks never emit a standalone "PART II" element."""
    texts = ["Item 7. MD&A", "Discussion.", "Item 1A. Risk Factors", "Risks."]
    categories = ["Title", "NarrativeText", "Title", "NarrativeText"]
    result = sectionise(texts, form_type="10-K", categories=categories)

    assert result.items[1] is not None
    assert result.items[1].key == "II-7"  # type: ignore[union-attr]
    assert result.items[3] is not None
    assert result.items[3].key == "I-1A"  # type: ignore[union-attr]


# ── canonical titles ────────────────────────────────────────
def test_canonical_title_overrides_the_filing_wording() -> None:
    """item_title is weighted 'A' in the lexical index, so it has to be the
    same string for every company or section-name queries rank unevenly."""
    assert ItemRef("II", "7").title("10-K").startswith("Management's Discussion")
    assert ItemRef("I", "1A").title("10-K") == "Risk Factors"


def test_title_tables_have_no_overlap_in_meaning() -> None:
    """10-K and 10-Q both define I-1 and II-1A; they mean different things."""
    assert ITEM_TITLES_10K["I-1"] == "Business"
    assert ITEM_TITLES_10Q["I-1"] == "Financial Statements"
    assert ITEM_TITLES_10K["I-1A"] == "Risk Factors"
    assert ITEM_TITLES_10Q["II-1A"] == "Risk Factors"


def test_every_element_is_assigned_or_explicitly_cover() -> None:
    """No element may silently vanish: an unassigned element is content that
    can never be retrieved."""
    texts, categories = _toc_then_body()
    result = sectionise(texts, form_type="10-K", categories=categories)
    assert len(result.items) == len(texts)


def test_headings_labelled_text_are_found() -> None:
    """Regression: the real Apple 10-K labels every Item heading "Text".

    The first implementation used an allowlist of {"Title", "Header",
    "NarrativeText", "UncategorizedText"} and therefore found *zero* headings
    in a real filing -- 543 elements all fell through to COVER, and every
    filtered query would have returned nothing while every unit test passed.
    The category check is now a denylist for exactly this reason.
    """
    texts = [
        "Item 1. Business",
        "We design and sell smartphones.",
        "Item 1A. Risk Factors",
        "Risks.",
    ]
    categories = ["Text", "NarrativeText", "Text", "NarrativeText"]
    result = sectionise(texts, form_type="10-K", categories=categories)

    assert [h.item.key for h in result.headings] == ["I-1", "I-1A"]
    assert result.items[1] is not None
    assert result.items[1].key == "I-1"  # type: ignore[union-attr]


def test_unknown_category_labels_do_not_hide_headings() -> None:
    """A future unstructured release may invent a new label; a heading under
    it must still be found rather than silently dropped."""
    result = sectionise(
        ["Item 7. MD&A", "Discussion."],
        form_type="10-K",
        categories=["SomeFutureCategory", "NarrativeText"],
    )
    assert [h.item.key for h in result.headings] == ["II-7"]


def test_categories_filter_prevents_narrative_false_positives() -> None:
    """A Table whose first cell reads "Item 1A" is not a section heading."""
    texts = ["Item 7. MD&A", "Item 1A", "Some prose."]
    categories = ["Title", "Table", "NarrativeText"]
    result = sectionise(texts, form_type="10-K", categories=categories)
    assert [h.item.number for h in result.headings] == ["7"]
