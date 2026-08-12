"""Assign every element of a filing to the Item it belongs to.

This is the part of ingestion that decides whether filtered retrieval works.
`unstructured` hands back a flat stream of elements with no notion of "Item 7A",
so without this step a chunk cannot be filtered to MD&A, and a chunk that
straddles the Item 1A/1B boundary mixes risk factors with staff comments.

Two problems make it harder than a regex:

1. **The table of contents.** Every 10-K opens with a TOC listing every Item
   heading verbatim. A naive scan therefore "finds" all 20 Items in the first
   two pages and assigns the entire body to Item 16. We detect the TOC run and
   skip it.

2. **10-Q Item numbers repeat.** Part I Item 1 is *Financial Statements*; Part
   II Item 1 is *Legal Proceedings*. Keying on the number alone silently merges
   two unrelated sections, so Part is tracked and forms half the key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

FormType = Literal["10-K", "10-Q"]

COVER = "COVER"

# "ITEM 7A." / "Item 1A - Risk Factors" / "ITEM 15 EXHIBITS" / "Item 6 [Reserved]"
# The trailing text is captured but never trusted: filings word their own
# headings inconsistently, so the canonical title comes from the tables below.
_ITEM_RE = re.compile(
    r"""^\s*
    (?:PART\s+[IV]+\s*[-–—:.]?\s*)?      # some filings inline the Part
    ITEM\s*
    (?P<num>\d{1,2})\s*
    (?P<suffix>[A-C])?
    \s*[.\-–—:)]?\s*
    (?P<title>.{0,150})$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PART_RE = re.compile(r"^\s*PART\s+(?P<part>I{1,3}V?|IV)\b\s*[.\-–—:]?\s*$", re.IGNORECASE)

# Canonical titles. Using our own table rather than the filing's wording keeps
# `item_title` stable across companies -- which matters because it is weighted
# 'A' in the lexical index, so inconsistent wording would make section-name
# queries rank differently per company.
ITEM_TITLES_10K: dict[str, str] = {
    "I-1": "Business",
    "I-1A": "Risk Factors",
    "I-1B": "Unresolved Staff Comments",
    "I-1C": "Cybersecurity",
    "I-2": "Properties",
    "I-3": "Legal Proceedings",
    "I-4": "Mine Safety Disclosures",
    "II-5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "II-6": "Reserved",
    "II-7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "II-7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "II-8": "Financial Statements and Supplementary Data",
    "II-9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "II-9A": "Controls and Procedures",
    "II-9B": "Other Information",
    "II-9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "III-10": "Directors, Executive Officers and Corporate Governance",
    "III-11": "Executive Compensation",
    "III-12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "III-13": "Certain Relationships and Related Transactions, and Director Independence",
    "III-14": "Principal Accountant Fees and Services",
    "IV-15": "Exhibits, Financial Statement Schedules",
    "IV-16": "Form 10-K Summary",
}

ITEM_TITLES_10Q: dict[str, str] = {
    "I-1": "Financial Statements",
    "I-2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk",
    "I-4": "Controls and Procedures",
    "II-1": "Legal Proceedings",
    "II-1A": "Risk Factors",
    "II-2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "II-3": "Defaults Upon Senior Securities",
    "II-4": "Mine Safety Disclosures",
    "II-5": "Other Information",
    "II-6": "Exhibits",
}

# Which Part an Item lives in, for filings that never emit a standalone
# "PART II" heading (common in 10-Ks that run Parts together).
_DEFAULT_PART_10K = {
    **dict.fromkeys(("1", "1A", "1B", "1C", "2", "3", "4"), "I"),
    **dict.fromkeys(("5", "6", "7", "7A", "8", "9", "9A", "9B", "9C"), "II"),
    **dict.fromkeys(("10", "11", "12", "13", "14"), "III"),
    **dict.fromkeys(("15", "16"), "IV"),
}

# Table-of-contents detection.
#
# Position alone is not a reliable signal -- how far into the element stream a
# TOC ends depends on the filing's length and on how many elements the
# partitioner emits per page. *Density* is reliable: a TOC is a run of Item
# headings with almost nothing between them, whereas body headings are
# separated by pages of prose. So we look for a dense run near the front.
TOC_WINDOW_FRACTION = 0.25  # the run must *start* within this leading fraction
TOC_MAX_GAP = 4  # elements between consecutive headings inside a TOC run
TOC_MIN_ITEMS = 5  # distinct Items required before we call a run a TOC

# Element categories that definitionally cannot be a section heading.
#
# This is a denylist rather than an allowlist, and the distinction is not
# cosmetic: an allowlist of {"Title", "Header", ...} failed silently on the
# very first real filing, because `unstructured`'s HTML partitioner labels
# Apple's Item headings "Text" -- a category no reasonable allowlist would have
# guessed. Failing closed on an unexpected label means zero headings, which
# means every chunk lands in COVER and every filtered query returns nothing.
# The anchored regex and the length guard do the real discrimination; the
# category check only has to exclude things that cannot contain a heading at
# all, and that set is small, stable, and knowable.
NON_HEADING_CATEGORIES = frozenset(
    {"Table", "Image", "PageBreak", "Footer", "Address", "EmailAddress", "FigureCaption"}
)


@dataclass(frozen=True)
class ItemRef:
    """An Item, keyed unambiguously by Part."""

    part: str  # "I" | "II" | "III" | "IV"
    number: str  # "1", "1A", "7A"

    @property
    def key(self) -> str:
        return f"{self.part}-{self.number}"

    def title(self, form_type: FormType) -> str:
        table = ITEM_TITLES_10K if form_type == "10-K" else ITEM_TITLES_10Q
        return table.get(self.key, f"Item {self.number}")


@dataclass
class Heading:
    """A candidate Item heading found at `index` in the element stream."""

    index: int
    item: ItemRef
    raw: str


@dataclass
class SectionAssignment:
    """Result of sectionising: one Item (or COVER) per element index."""

    items: list[ItemRef | None]
    headings: list[Heading]
    toc_end_index: int = 0
    section_path: list[list[str]] = field(default_factory=list)


def normalise_item_number(num: str, suffix: str | None) -> str:
    """`("7", "a") -> "7A"`. Suffix case is not meaningful in filings."""
    return f"{num}{suffix.upper()}" if suffix else num


def parse_item_heading(text: str) -> tuple[str, str] | None:
    """Return `(number, raw_title)` if `text` looks like an Item heading.

    Rejects prose that merely mentions an Item ("as described in Item 1A
    above"), because the regex is anchored and the leading text would not
    match.
    """
    stripped = " ".join(text.split())
    if not stripped or len(stripped) > 200:
        return None
    m = _ITEM_RE.match(stripped)
    if not m:
        return None
    number = normalise_item_number(m.group("num"), m.group("suffix"))
    title = m.group("title").strip(" .:-–—")
    return number, title


def parse_part_heading(text: str) -> str | None:
    m = _PART_RE.match(" ".join(text.split()))
    return m.group("part").upper() if m else None


def _detect_toc_end(headings: list[Heading], total_elements: int) -> int:
    """Return the element index after which body content begins.

    Finds the first *dense run* of Item headings -- consecutive headings
    separated by at most TOC_MAX_GAP elements. A table of contents is exactly
    that shape: twenty headings back to back with nothing in between. The body
    never is, because every real Item is followed by paragraphs before the next
    heading appears.

    Returns 0 (no TOC) when the run is too short, or starts too late in the
    document to be a TOC. Both guards matter: an excerpt containing only a
    couple of sections must not have its first section eaten.
    """
    if total_elements == 0 or not headings:
        return 0

    # Walk the first run of headings that are packed tightly together, and stop
    # at the first *repeat*. The repeat is the sharper of the two signals: a
    # TOC lists each Item once in order, and then the body immediately restarts
    # the same sequence from Item 1. Density alone would run straight through
    # that boundary whenever the body's first heading follows the TOC's last
    # one closely, swallowing Item 1.
    seen = {headings[0].item.number}
    run_end = 0
    for i in range(1, len(headings)):
        if headings[i].index - headings[i - 1].index > TOC_MAX_GAP:
            break
        if headings[i].item.number in seen:
            break
        seen.add(headings[i].item.number)
        run_end = i

    run = headings[: run_end + 1]
    if len({h.item.number for h in run}) < TOC_MIN_ITEMS:
        return 0
    if run[0].index > max(1, int(total_elements * TOC_WINDOW_FRACTION)):
        return 0
    return run[-1].index + 1


def sectionise(
    texts: list[str],
    *,
    form_type: FormType,
    categories: list[str] | None = None,
) -> SectionAssignment:
    """Assign each element in `texts` to an Item.

    `categories` are the unstructured element categories (``Title``,
    ``NarrativeText``, ...). When supplied, only heading-ish elements are
    considered as Item headings, which removes the main false-positive source:
    a narrative paragraph that happens to begin "Item 1A."
    """
    headings: list[Heading] = []
    current_part: str | None = None
    default_part = _DEFAULT_PART_10K if form_type == "10-K" else {}

    for i, raw in enumerate(texts):
        if (part := parse_part_heading(raw)) is not None:
            current_part = part
            continue

        if categories is not None and categories[i] in NON_HEADING_CATEGORIES:
            continue

        parsed = parse_item_heading(raw)
        if parsed is None:
            continue

        number, _title = parsed
        part = current_part or default_part.get(number) or "I"
        headings.append(Heading(index=i, item=ItemRef(part=part, number=number), raw=raw.strip()))

    toc_end = _detect_toc_end(headings, len(texts))
    body_headings = [h for h in headings if h.index >= toc_end]

    items: list[ItemRef | None] = [None] * len(texts)
    current: ItemRef | None = None
    heading_at = {h.index: h.item for h in body_headings}
    for i in range(len(texts)):
        if i in heading_at:
            current = heading_at[i]
        items[i] = current

    return SectionAssignment(items=items, headings=body_headings, toc_end_index=toc_end)
