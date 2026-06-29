"""
example_runthrough.py
Citibank Applied Research Project - single-document example runner.

This keeps the original fake-transcript demo, but now reuses the shared
pipeline helpers that power the batch report runner in run_reports.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from report_pipeline import (
    OUTPUTS_DIR,
    ReportSpec,
    build_doc_params,
    build_user_message,
    call_llm,
    consistency_check,
    utc_run_id,
)

FAKE_TRANSCRIPT = """META PLATFORMS Q3 2024 EARNINGS CALL TRANSCRIPT (EXAMPLE - FICTIONAL)
October 30, 2024

MARK ZUCKERBERG (CEO): Revenue came in at $40.6 billion, up 19% year-over-year,
which was at the top end of our guidance range of $38.5 to $41 billion. We are
seeing real momentum from our AI investments, particularly in ads ranking and
content recommendations, where our Llama-based models have driven measurable
improvements in engagement and click-through rates that are exceeding our internal
targets. Llama adoption continues to accelerate. We now have over 400 million
monthly active users on Meta AI. On the risk side, we continue to face uncertainty
around data privacy enforcement in Europe, and the outcome of current proceedings
could affect how we operate in the region. This remains a meaningful source of
uncertainty.

SUSAN LI (CFO): Revenue for Q3 2024 was $40.6 billion, a 19% increase
year-over-year. Earnings per share were $6.03, compared to $4.39 in Q3 2023,
an increase of 37%. Operating income was $17.4 billion, representing an operating
margin of 43%. For Q4 2024, we expect revenue to be in the range of $45 to $48
billion. We are raising our full-year 2024 capital expenditure guidance to $38 to
$40 billion. We expect capital expenditures to increase meaningfully again in 2025.
We will not slow investment in AI where we see clear returns.

ANALYST: On European regulatory risk - can you size the potential revenue impact?

SUSAN LI: We're not in a position to quantify it precisely. Europe represents a
meaningful portion of our global revenue, so any structural change to how we
operate there would be material."""

EXAMPLE_REPORT = ReportSpec(
    issuer="example",
    company="Meta Platforms",
    ticker="META",
    sector="Technology",
    report_type="Earnings Call Transcript",
    fiscal_period="Q3 2024",
    report_date="2024-10-30",
    source_pdf=Path("example"),
    document_id="META_Q3_2024_EXAMPLE",
)


def main() -> None:
    run_id = utc_run_id()
    output_dir = OUTPUTS_DIR / "example"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_params = build_doc_params(
        EXAMPLE_REPORT,
        FAKE_TRANSCRIPT,
        hold_upper=0.15,
        hold_lower=-0.15,
    )
    user_message = build_user_message(doc_params)

    print("=" * 60)
    print("LLM Analysis Pipeline - DeepSeek Example")
    print(f"Document: {EXAMPLE_REPORT.ticker} {EXAMPLE_REPORT.fiscal_period}")
    print("=" * 60)

    print("\n[1/2] Running single analysis call...")
    output = call_llm(user_message, doc_params, EXAMPLE_REPORT, run_id, cached_input=False)
    result = output["result"]
    cost_log = output["cost_log"]

    print("\nRESULT")
    print(f"  Signal:    {result['signal']['direction']}")
    print(f"  Score:     {result['sentiment']['score']}")
    print(f"  Rationale: {result['sentiment']['rationale']}")

    result_target = output_dir / "example_result.json"
    result_target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "cost_log.jsonl").write_text(
        json.dumps(cost_log, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("\n[2/2] Running consistency check (3 runs)...")
    consistency = consistency_check(
        user_message,
        doc_params,
        EXAMPLE_REPORT,
        run_id,
        n_runs=3,
    )
    (output_dir / "consistency_log.json").write_text(
        json.dumps(consistency["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  Example result saved to {result_target}")
    print(f"  Estimated first-run cost: ${cost_log['estimated_cost_usd']:.5f}")
    print("Done.")


if __name__ == "__main__":
    main()
