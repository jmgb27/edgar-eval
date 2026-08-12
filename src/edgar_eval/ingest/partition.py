"""Turn filing HTML into a stream of elements.

`Partitioner` is a protocol rather than a direct call to `unstructured` for a
reason worth stating in the README: Unstructured is heavier than this job
strictly needs and has no concept of "Item 7A", so an lxml + regex sectioniser
is both faster and more accurate on 10-K structure. What Unstructured *does*
earn its place on is table extraction -- `metadata.text_as_html` is genuinely
good, and tables are the hard part of a financial filing.

Keeping the boundary explicit means swapping the implementation is one class,
and the swap can be measured against the eval set rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from edgar_eval.ingest.edgar import strip_noise
from edgar_eval.logging import get_logger

log = get_logger(__name__)


@dataclass
class PartitionedElement:
    """One element, flattened to just what the rest of the pipeline uses."""

    text: str
    category: str
    table_html: str | None = None
    page_number: int | None = None
    raw: Any = None


class Partitioner(Protocol):
    def partition(self, html: str) -> list[PartitionedElement]: ...


class UnstructuredHtmlPartitioner:
    """`unstructured.partition.html`, configured for SEC filings.

    Deliberately *not* the PDF path. EDGAR serves the authoritative filing as
    HTML, and `partition_pdf(strategy="hi_res")` would run layout detection per
    page -- minutes of CPU and multi-gigabyte RSS on a 300-page 10-K, for a
    worse result than parsing the HTML we already have.
    """

    def __init__(self, *, infer_table_structure: bool = True) -> None:
        self._infer_table_structure = infer_table_structure

    def partition(self, html: str) -> list[PartitionedElement]:
        from unstructured.partition.html import partition_html

        cleaned = strip_noise(html)
        log.debug("partition.start", input_chars=len(html), cleaned_chars=len(cleaned))

        elements = partition_html(
            text=cleaned,
            infer_table_structure=self._infer_table_structure,
            # Chunking happens later, per Item. Letting the partitioner chunk
            # here would merge across Item boundaries before we can stop it.
            chunking_strategy=None,
        )

        out: list[PartitionedElement] = []
        for element in elements:
            text = str(element).strip()
            if not text:
                continue
            metadata = getattr(element, "metadata", None)
            out.append(
                PartitionedElement(
                    text=text,
                    category=type(element).__name__,
                    table_html=getattr(metadata, "text_as_html", None),
                    page_number=getattr(metadata, "page_number", None),
                    raw=element,
                )
            )
        log.debug("partition.done", elements=len(out))
        return out
