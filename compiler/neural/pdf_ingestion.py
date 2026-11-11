"""
=============================================================================
MUDRAFORGE STATUTORY PDF & DOCUMENT INGESTION ENGINE
=============================================================================

Automated layout-aware extraction of statutory text from raw government PDFs
and documents without human handholding.

Features:
- PyMuPDF (fitz) layout-aware block extraction
- Devanagari / bilingual stream separation (isolates English legislative text)
- Hyphenated line-break repair (e.g., 'ex- \n ceeding' -> 'exceeding')
- Boilerplate & printer watermark suppression (Dte. of Printing, digital signatures)
- Statutory metadata extraction (Circular/Notification number, date, authority)
- Sentence & clause segmentation for neural parsing
"""

import os
import re
import sys
import fitz # PyMuPDF

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Regex patterns for statutory metadata
CIRCULAR_REGEX = re.compile(
    r"(?:Circular\s+No\.?|Notification\s+No\.?|S\.O\.)\s*([0-9A-Za-z\/\-\(\)\.]+)",
    re.IGNORECASE
)
DATE_REGEX = re.compile(
    r"(?:Dated|New\s+Delhi,\s+the)\s*[:–-]?\s*([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+[0-9]{4}|[0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4})",
    re.IGNORECASE
)
AUTHORITY_REGEX = re.compile(
    r"(Central\s+Board\s+of\s+Direct\s+Taxes|Central\s+Board\s+of\s+Indirect\s+Taxes\s+and\s+Customs|Ministry\s+of\s+Finance|Ministry\s+of\s+Micro,\s+Small\s+and\s+Medium\s+Enterprises)",
    re.IGNORECASE
)

# Boilerplate patterns to suppress
BOILERPLATE_PATTERNS = [
    re.compile(r"Uploaded\s+by\s+Dte\.\s+of\s+Printing.*", re.IGNORECASE),
    re.compile(r"Digitally\s+signed\s+by.*", re.IGNORECASE),
    re.compile(r"PUBLISHED\s+BY\s+AUTHORITY.*", re.IGNORECASE),
    re.compile(r"THE\s+GAZETTE\s+OF\s+INDIA\s*:\s*EXTRAORDINARY.*", re.IGNORECASE),
    re.compile(r"REGD\.\s+No\.\s+D\.\s*L\.-[0-9\/]+", re.IGNORECASE),
    re.compile(r"xxxGID[A-Z]+xxx", re.IGNORECASE),
]


def is_devanagari_block(text: str) -> bool:
    """Returns True if the block contains predominantly Hindi / Devanagari characters."""
    total_chars = len(text.strip())
    if total_chars == 0:
        return False
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    # If more than 30% of the characters are Devanagari, treat as non-English
    return (devanagari_chars / total_chars) > 0.30


def repair_hyphenation(text: str) -> str:
    """Repairs split words across line-breaks like 'pur- \n chase' -> 'purchase'."""
    # Pattern: letters followed by hyphen, newline, optional whitespace, and lowercase letters
    repaired = re.sub(r"([A-Za-z]+)-\s*\n\s*([a-z]+)", r"\1\2", text)
    # Replace soft hyphens
    repaired = repaired.replace("\xad", "")
    return repaired


def clean_block_text(text: str) -> str:
    """Cleans a single text block: filters out Hindi lines, repairs hyphenation, trims boilerplate."""
    # 1. Filter out any lines containing Devanagari script
    english_lines = []
    for line in text.split("\n"):
        if not re.search(r"[\u0900-\u097F]", line) and line.strip():
            english_lines.append(line.strip())

    if not english_lines:
        return ""

    joined = "\n".join(english_lines)
    t = repair_hyphenation(joined)
    for bp in BOILERPLATE_PATTERNS:
        t = bp.sub("", t)

    # Collapse multiple whitespace/newlines within paragraphs to single space
    lines = [line.strip() for line in t.split("\n") if line.strip()]
    return " ".join(lines)


class StatutoryDocumentIngestor:
    """
    Ingests and normalizes official statutory documents (PDF, text)
    for the Mudra Statutory AST Compiler.
    """

    def __init__(self):
        pass

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Parses a PDF file using PyMuPDF and extracts clean statutory text and metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        raw_blocks = []
        english_blocks = []
        full_text_parts = []

        raw_header_text = ""

        for page_idx, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for b in blocks:
                # b: (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) >= 5:
                    block_text = b[4].strip()
                    if not block_text:
                        continue

                    raw_blocks.append({
                        "page": page_idx,
                        "bbox": b[:4],
                        "text": block_text
                    })

                    # If page 0, collect first few blocks for metadata extraction
                    if page_idx == 0 and len(raw_header_text) < 1000:
                        raw_header_text += " " + block_text

                    # Discard Devanagari blocks
                    if is_devanagari_block(block_text):
                        continue

                    cleaned = clean_block_text(block_text)
                    if cleaned:
                        english_blocks.append({
                            "page": page_idx,
                            "bbox": b[:4],
                            "text": cleaned
                        })
                        full_text_parts.append(cleaned)

        full_cleaned_text = "\n\n".join(full_text_parts)

        # Extract metadata
        metadata = self._extract_metadata(raw_header_text, full_cleaned_text)

        # Segment into candidate statutory clauses
        clauses = self._segment_clauses(full_cleaned_text)

        return {
            "source_file": os.path.basename(pdf_path),
            "page_count": len(doc),
            "total_blocks": len(raw_blocks),
            "english_blocks_count": len(english_blocks),
            "metadata": metadata,
            "full_text": full_cleaned_text,
            "clauses": clauses
        }

    def ingest_text(self, text: str, source_name: str = "raw_text") -> dict:
        """
        Ingests plain text string.
        """
        cleaned = clean_block_text(text)
        metadata = self._extract_metadata(text, cleaned)
        clauses = self._segment_clauses(cleaned)
        return {
            "source_file": source_name,
            "page_count": 1,
            "total_blocks": 1,
            "english_blocks_count": 1,
            "metadata": metadata,
            "full_text": cleaned,
            "clauses": clauses
        }

    def _extract_metadata(self, header_text: str, full_text: str) -> dict:
        combined = header_text + " " + full_text[:2000]

        circ_match = CIRCULAR_REGEX.search(combined)
        circular_no = circ_match.group(0).strip() if circ_match else "UNKNOWN_INSTRUMENT"

        date_match = DATE_REGEX.search(combined)
        date_str = date_match.group(1).strip() if date_match else "UNKNOWN_DATE"

        auth_match = AUTHORITY_REGEX.search(combined)
        authority = auth_match.group(1).strip() if auth_match else "UNKNOWN_AUTHORITY"

        # Check for statutory sections referenced
        sections_found = list(set(re.findall(r"section\s+([0-9]{1,3}[A-Z]*(?:\([0-9A-Za-z]+\))*)", combined, re.IGNORECASE)))

        return {
            "instrument_reference": circular_no,
            "issuing_date": date_str,
            "issuing_authority": authority,
            "referenced_sections": sections_found
        }

    def _segment_clauses(self, full_text: str) -> list[str]:
        """
        Segments text into individual statutory clauses based on paragraph,
        sub-section numbering, or punctuation boundaries.
        """
        # Split on paragraph breaks or numbered clauses: (i), (ii), 1., 2., etc.
        raw_paragraphs = re.split(r"\n\n+|(?=\([ivx0-9]+\)\s+)|(?=[0-9]+\.\s+)", full_text)
        clauses = []
        for p in raw_paragraphs:
            p_clean = p.strip()
            # Retain paragraphs that have substantial content (> 20 chars)
            if len(p_clean) > 20:
                clauses.append(p_clean)
        return clauses


if __name__ == "__main__":
    import json
    import sys

    ingestor = StatutoryDocumentIngestor()
    test_pdf = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    if os.path.exists(test_pdf):
        result = ingestor.ingest_pdf(test_pdf)
        print("--- EXTRACTED METADATA ---")
        print(json.dumps(result["metadata"], indent=2))
        print(f"--- TOTAL CLAUSES EXTRACTED: {len(result['clauses'])} ---")
        for i, c in enumerate(result["clauses"][:3]):
            print(f"[{i+1}] {c[:140]}...")
    else:
        print(f"Test PDF not found at {test_pdf}")
