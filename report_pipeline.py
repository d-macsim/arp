from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs"
ENV_FILE = BASE_DIR / ".env"
PROMPT_TEMPLATE_FILE = PROMPTS_DIR / "llm_analysis_prompt_template.md"

DEEPSEEK_INPUT_PRICE_PER_M = 0.27
DEEPSEEK_CACHED_PRICE_PER_M = 0.07
DEEPSEEK_OUTPUT_PRICE_PER_M = 1.10

MIN_EXTRACTED_TEXT_CHARS = 2000
SPEAKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9 .,&'()/:-]{2,}:")
SPEAKER_TITLE_PATTERN = re.compile(
    r"\b(Co-CEO|Chief Financial Officer|Chief Executive Officer|Vice President|Director|Operator)\b"
)
PAGE_NUMBER_PATTERN = re.compile(r"^\d+$")

load_dotenv(dotenv_path=ENV_FILE)


@dataclass(frozen=True)
class ReportSpec:
    issuer: str
    company: str
    ticker: str
    sector: str
    report_type: str
    fiscal_period: str
    report_date: str
    source_pdf: Path
    document_id: str

    @property
    def output_stem(self) -> str:
        return sanitize_filename(self.document_id)


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    page_count: int
    extracted_characters: int
    warnings: list[str]
    extractor_used: str


def report_metadata(report: ReportSpec) -> dict[str, Any]:
    return {
        "issuer": report.issuer,
        "company": report.company,
        "ticker": report.ticker,
        "sector": report.sector,
        "report_type": report.report_type,
        "fiscal_period": report.fiscal_period,
        "report_date": report.report_date,
        "source_pdf": str(report.source_pdf),
        "document_id": report.document_id,
    }


def load_prompt_sections() -> tuple[str, str]:
    template_text = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    fenced_blocks = [
        block.strip()
        for index, block in enumerate(template_text.split("```"))
        if index % 2 == 1
    ]
    if len(fenced_blocks) < 2:
        raise ValueError(
            f"Expected at least two fenced code blocks in {PROMPT_TEMPLATE_FILE}"
        )
    return fenced_blocks[0], fenced_blocks[1]


SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE = load_prompt_sections()


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("_") or "document"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_dumps_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True) + "\n"


def load_manifest(manifest_path: Path) -> list[ReportSpec]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = raw["reports"] if isinstance(raw, dict) else raw
    loaded_reports = []

    for entry in reports:
        source_pdf = Path(entry["source_pdf"])
        if not source_pdf.is_absolute():
            source_pdf = (BASE_DIR / source_pdf).resolve()
        loaded_reports.append(
            ReportSpec(
                issuer=entry["issuer"],
                company=entry["company"],
                ticker=entry["ticker"],
                sector=entry["sector"],
                report_type=entry["report_type"],
                fiscal_period=entry["fiscal_period"],
                report_date=entry["report_date"],
                source_pdf=source_pdf,
                document_id=entry["document_id"],
            )
        )

    return loaded_reports


def build_user_message(params: dict[str, Any]) -> str:
    placeholder_map = {
        "COMPANY": params["company"],
        "TICKER": params["ticker"],
        "SECTOR": params["sector"],
        "REPORT_TYPE": params["report_type"],
        "REPORT_DATE": params["report_date"],
        "FISCAL_PERIOD": params["fiscal_period"],
        "REPORT_TEXT": params["report_text"],
        "HOLD_UPPER": params["hold_upper"],
        "HOLD_LOWER": params["hold_lower"],
    }

    user_message = USER_MESSAGE_TEMPLATE
    for key, value in placeholder_map.items():
        user_message = user_message.replace(f"{{{{{key}}}}}", str(value))
    return user_message


def build_doc_params(
    report: ReportSpec,
    report_text: str,
    hold_upper: float,
    hold_lower: float,
) -> dict[str, Any]:
    return {
        "company": report.company,
        "ticker": report.ticker,
        "sector": report.sector,
        "report_type": report.report_type,
        "report_date": report.report_date,
        "fiscal_period": report.fiscal_period,
        "report_text": report_text,
        "hold_upper": hold_upper,
        "hold_lower": hold_lower,
    }


def _extract_text_with_pypdf(pdf_path: Path) -> tuple[list[str], int]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    return page_texts, len(reader.pages)


def _extract_text_with_pdfplumber(pdf_path: Path) -> tuple[list[str], int]:
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_texts = [(page.extract_text() or "").strip() for page in pdf.pages]
        return page_texts, len(pdf.pages)


def _extract_text_from_html(html_path: Path) -> tuple[list[str], int]:
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._skip = False
            self.chunks: list[str] = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in ("script", "style", "noscript"):
                self._skip = True

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "noscript"):
                self._skip = False

        def handle_data(self, data: str) -> None:
            if not self._skip:
                text = data.strip()
                if text:
                    self.chunks.append(text)

    parser = _TextExtractor()
    parser.feed(html_path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(parser.chunks)
    return [text], 1


def _normalize_margin_candidate(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _identify_repeated_margin_lines(page_texts: list[str]) -> tuple[set[str], set[str]]:
    start_counter: Counter[str] = Counter()
    end_counter: Counter[str] = Counter()

    for page_text in page_texts:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if not lines:
            continue
        for line in lines[:2]:
            start_counter[_normalize_margin_candidate(line)] += 1
        for line in lines[-2:]:
            end_counter[_normalize_margin_candidate(line)] += 1

    repeated_starts = {line for line, count in start_counter.items() if count >= 2}
    repeated_ends = {line for line, count in end_counter.items() if count >= 2}
    return repeated_starts, repeated_ends


def _merge_wrapped_lines(page_texts: list[str]) -> str:
    repeated_starts, repeated_ends = _identify_repeated_margin_lines(page_texts)
    merged_blocks: list[str] = []
    paragraph = ""

    for page_text in page_texts:
        raw_lines = page_text.splitlines()
        filtered_lines: list[str] = []

        for index, raw_line in enumerate(raw_lines):
            line = raw_line.strip()
            if not line:
                filtered_lines.append("")
                continue

            normalized_line = _normalize_margin_candidate(line)
            if index < 2 and normalized_line in repeated_starts:
                continue
            if len(raw_lines) - index <= 2 and normalized_line in repeated_ends:
                continue
            if PAGE_NUMBER_PATTERN.match(line):
                continue

            filtered_lines.append(re.sub(r"\s+", " ", line))

        for line in filtered_lines:
            if not line:
                if paragraph:
                    merged_blocks.append(paragraph.strip())
                    paragraph = ""
                continue

            if SPEAKER_PATTERN.match(line):
                if paragraph:
                    merged_blocks.append(paragraph.strip())
                paragraph = line
                continue

            if not paragraph:
                paragraph = line
                continue

            if paragraph.endswith("-"):
                paragraph = paragraph[:-1] + line
            else:
                paragraph = f"{paragraph} {line}"

        if paragraph:
            merged_blocks.append(paragraph.strip())
            paragraph = ""

    normalized_text = "\n\n".join(block for block in merged_blocks if block)
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
    return normalized_text.strip()


def extract_doc_text(pdf_path: Path) -> ExtractionResult:
    warnings: list[str] = []

    if pdf_path.suffix.lower() == ".html":
        page_texts, page_count = _extract_text_from_html(pdf_path)
        extractor_used = "html.parser"
        cleaned_text = _merge_wrapped_lines(page_texts)
        extracted_characters = len(cleaned_text)
        if extracted_characters < MIN_EXTRACTED_TEXT_CHARS:
            raise ValueError(
                f"Extracted text from {pdf_path.name} is too short ({extracted_characters} characters)"
            )
        if "Earnings Call" not in cleaned_text and "Earnings Call Transcripts" not in cleaned_text:
            warnings.append("missing_keyword:Earnings Call")
        return ExtractionResult(
            text=cleaned_text,
            page_count=page_count,
            extracted_characters=extracted_characters,
            warnings=warnings,
            extractor_used=extractor_used,
        )

    extractor_used = "pypdf"
    page_texts, page_count = _extract_text_with_pypdf(pdf_path)
    pypdf_text_length = len("".join(page_texts).strip())

    need_fallback = pypdf_text_length < MIN_EXTRACTED_TEXT_CHARS or not any(
        page_texts
    )

    if need_fallback:
        fallback_page_texts, fallback_page_count = _extract_text_with_pdfplumber(pdf_path)
        fallback_text_length = len("".join(fallback_page_texts).strip())
        if fallback_text_length > pypdf_text_length:
            page_texts = fallback_page_texts
            page_count = fallback_page_count
            extractor_used = "pdfplumber"
    else:
        fallback_page_texts, _ = _extract_text_with_pdfplumber(pdf_path)
        page_texts = [
            fallback_page_texts[index]
            if not page_text.strip() and index < len(fallback_page_texts)
            else page_text
            for index, page_text in enumerate(page_texts)
        ]

    if page_count == 0:
        raise ValueError(f"No pages found in PDF: {pdf_path}")

    cleaned_text = _merge_wrapped_lines(page_texts)
    extracted_characters = len(cleaned_text)

    if extracted_characters < MIN_EXTRACTED_TEXT_CHARS:
        raise ValueError(
            f"Extracted text from {pdf_path.name} is too short ({extracted_characters} characters)"
        )

    if "Earnings Call" not in cleaned_text and "Earnings Call Transcripts" not in cleaned_text:
        warnings.append("missing_keyword:Earnings Call")
    has_speaker_labels = any(
        SPEAKER_PATTERN.match(block) for block in cleaned_text.split("\n\n")
    )
    has_speaker_titles = bool(SPEAKER_TITLE_PATTERN.search(cleaned_text))
    if not has_speaker_labels and not has_speaker_titles:
        warnings.append("missing_pattern:speaker_labels")

    return ExtractionResult(
        text=cleaned_text,
        page_count=page_count,
        extracted_characters=extracted_characters,
        warnings=warnings,
        extractor_used=extractor_used,
    )


def strip_json_fences(raw_text: str) -> str:
    clean_text = raw_text.strip()
    if clean_text.startswith("```"):
        parts = clean_text.split("```")
        if len(parts) > 1:
            clean_text = parts[1]
        if clean_text.startswith("json"):
            clean_text = clean_text[4:]
        clean_text = clean_text.strip()
    return clean_text


def call_llm(
    user_message: str,
    doc_params: dict[str, Any],
    report: ReportSpec,
    run_id: str,
    cached_input: bool = False,
    model: str = "deepseek-chat",
) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in the environment or .env")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    elapsed = time.time() - start_time
    raw_text = response.choices[0].message.content or ""
    result = json.loads(strip_json_fences(raw_text))

    input_price = (
        DEEPSEEK_CACHED_PRICE_PER_M if cached_input else DEEPSEEK_INPUT_PRICE_PER_M
    )

    cost_log = {
        "run_id": run_id,
        "timestamp": utc_timestamp(),
        "status": "success",
        "document_id": report.document_id,
        "issuer": report.issuer,
        "company": report.company,
        "ticker": report.ticker,
        "fiscal_period": report.fiscal_period,
        "report_date": report.report_date,
        "source_pdf": str(report.source_pdf),
        "model": response.model,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "cached_input": cached_input,
        "estimated_cost_usd": round(
            (response.usage.prompt_tokens * input_price / 1_000_000)
            + (
                response.usage.completion_tokens
                * DEEPSEEK_OUTPUT_PRICE_PER_M
                / 1_000_000
            ),
            6,
        ),
        "latency_seconds": round(elapsed, 2),
        "signal": result["signal"]["direction"],
        "sentiment_score": result["sentiment"]["score"],
        "hold_upper": doc_params["hold_upper"],
        "hold_lower": doc_params["hold_lower"],
    }

    return {"result": result, "cost_log": cost_log}


def consistency_check(
    user_message: str,
    doc_params: dict[str, Any],
    report: ReportSpec,
    run_id: str,
    n_runs: int,
    model: str = "deepseek-chat",
) -> dict[str, Any]:
    signals = []
    scores = []
    cost_logs = []

    for index in range(n_runs):
        output = call_llm(
            user_message,
            doc_params,
            report,
            run_id,
            cached_input=True,
            model=model,
        )
        signals.append(output["result"]["signal"]["direction"])
        scores.append(output["result"]["sentiment"]["score"])
        cost_log = dict(output["cost_log"])
        cost_log["consistency_run_index"] = index + 1
        cost_logs.append(cost_log)

    summary = {
        "run_id": run_id,
        "document_id": report.document_id,
        "n_runs": n_runs,
        "signals": signals,
        "scores": scores,
        "signal_agreement": len(set(signals)) == 1,
        "score_range": round(max(scores) - min(scores), 3),
        "total_cost_usd": round(sum(item["estimated_cost_usd"] for item in cost_logs), 6),
    }
    return {"summary": summary, "cost_logs": cost_logs}


def build_result_payload(
    raw_result: dict[str, Any],
    report: ReportSpec,
    cost_log: dict[str, Any],
    extraction_target: Path,
    extraction_meta: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    payload = dict(raw_result)
    model_document_id = payload.get("document_id")
    payload["document_id"] = report.document_id
    if model_document_id and model_document_id != report.document_id:
        payload["model_document_id"] = model_document_id

    payload["report_metadata"] = report_metadata(report)
    payload["run_meta"] = {
        "run_id": cost_log["run_id"],
        "timestamp": cost_log["timestamp"],
        "model": cost_log["model"],
        "cached_input": cost_log["cached_input"],
        "source_pdf": str(report.source_pdf),
        "extracted_text_path": str(extraction_target),
    }
    if extraction_meta is not None:
        payload["extraction_meta"] = extraction_meta
    if warnings:
        payload["extraction_warnings"] = warnings
    return payload
