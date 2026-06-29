from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from report_pipeline import (
    BASE_DIR,
    OUTPUTS_DIR,
    ReportSpec,
    build_result_payload,
    build_doc_params,
    build_user_message,
    call_llm,
    consistency_check,
    extract_doc_text,
    json_dumps_line,
    load_manifest,
    sanitize_filename,
    utc_run_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run earnings call reports through the LLM pipeline."
    )
    parser.add_argument(
        "--issuer",
        default="netflix",
        help="Issuer name to process. Default: netflix",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE_DIR / "manifests" / "netflix_reports.json",
        help="Path to the report manifest JSON file.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing extracted text files instead of re-extracting PDFs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction and planning steps without calling the LLM API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of reports processed. Default: 0 means all.",
    )
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=0,
        help="Optional number of extra cached consistency runs per report.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name for DeepSeek's OpenAI-compatible API.",
    )
    parser.add_argument(
        "--hold-upper",
        type=float,
        default=0.15,
        help="BUY threshold. Default: 0.15",
    )
    parser.add_argument(
        "--hold-lower",
        type=float,
        default=-0.15,
        help="SELL threshold. Default: -0.15",
    )
    return parser.parse_args()


def ensure_output_dirs(issuer: str) -> dict[str, Path]:
    issuer_root = OUTPUTS_DIR / issuer
    dirs = {
        "root": issuer_root,
        "extracted": issuer_root / "extracted",
        "results": issuer_root / "results",
        "logs": issuer_root / "logs",
        "summary": issuer_root / "summary",
        "runs": issuer_root / "runs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def extracted_text_path(output_dirs: dict[str, Path], report: ReportSpec) -> Path:
    return output_dirs["extracted"] / f"{report.output_stem}.txt"


def result_path(output_dirs: dict[str, Path], report: ReportSpec) -> Path:
    return output_dirs["results"] / f"{report.output_stem}.json"


def first_run_log_path(output_dirs: dict[str, Path]) -> Path:
    return output_dirs["logs"] / "first_run_costs.jsonl"


def consistency_log_path(output_dirs: dict[str, Path]) -> Path:
    return output_dirs["logs"] / "consistency_costs.jsonl"


def consistency_summary_path(output_dirs: dict[str, Path], issuer: str) -> Path:
    return output_dirs["summary"] / f"{sanitize_filename(issuer)}_consistency_summary.json"


def first_run_summary_path(output_dirs: dict[str, Path], issuer: str) -> Path:
    return output_dirs["summary"] / f"{sanitize_filename(issuer)}_first_run_summary.csv"


def run_root_path(output_dirs: dict[str, Path], run_id: str) -> Path:
    return output_dirs["runs"] / run_id


def ensure_run_output_dirs(output_dirs: dict[str, Path], run_id: str) -> dict[str, Path]:
    run_root = run_root_path(output_dirs, run_id)
    dirs = {
        "root": run_root,
        "results": run_root / "results",
        "logs": run_root / "logs",
        "summary": run_root / "summary",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def archived_result_path(run_output_dirs: dict[str, Path], report: ReportSpec) -> Path:
    return run_output_dirs["results"] / f"{report.output_stem}.json"


def archived_first_run_log_path(run_output_dirs: dict[str, Path]) -> Path:
    return run_output_dirs["logs"] / "first_run_costs.jsonl"


def archived_consistency_log_path(run_output_dirs: dict[str, Path]) -> Path:
    return run_output_dirs["logs"] / "consistency_costs.jsonl"


def archived_first_run_summary_path(
    run_output_dirs: dict[str, Path], issuer: str, run_id: str
) -> Path:
    return run_output_dirs["summary"] / f"{sanitize_filename(issuer)}_first_run_summary_{run_id}.csv"


def archived_consistency_summary_path(
    run_output_dirs: dict[str, Path], issuer: str, run_id: str
) -> Path:
    return run_output_dirs["summary"] / f"{sanitize_filename(issuer)}_consistency_summary_{run_id}.json"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_dumps_line(payload))


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "run_id",
        "document_id",
        "issuer",
        "company",
        "ticker",
        "fiscal_period",
        "report_date",
        "source_pdf",
        "model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input",
        "estimated_cost_usd",
        "latency_seconds",
        "signal",
        "sentiment_score",
        "status",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_extracted_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def process_report(
    report: ReportSpec,
    output_dirs: dict[str, Path],
    run_output_dirs: dict[str, Path],
    run_id: str,
    args: argparse.Namespace,
    first_report: bool,
) -> dict[str, Any]:
    report_output = {
        "report": report,
        "success": False,
        "dry_run": args.dry_run,
        "warnings": [],
        "first_run_log": None,
        "consistency": None,
    }
    extraction_target = extracted_text_path(output_dirs, report)

    if args.skip_extract:
        if not extraction_target.exists():
            raise FileNotFoundError(
                f"Missing extracted text for {report.document_id}: {extraction_target}"
            )
        report_text = load_extracted_text(extraction_target)
    else:
        extraction = extract_doc_text(report.source_pdf)
        extraction_target.write_text(extraction.text, encoding="utf-8")
        report_text = extraction.text
        report_output["warnings"].extend(extraction.warnings)
        report_output["extraction"] = {
            "page_count": extraction.page_count,
            "extracted_characters": extraction.extracted_characters,
            "extractor_used": extraction.extractor_used,
            "target_path": str(extraction_target),
        }

    doc_params = build_doc_params(
        report,
        report_text,
        hold_upper=args.hold_upper,
        hold_lower=args.hold_lower,
    )

    if args.dry_run:
        report_output["success"] = True
        report_output["planned_result_path"] = str(result_path(output_dirs, report))
        report_output["doc_params"] = {
            key: value for key, value in doc_params.items() if key != "report_text"
        }
        return report_output

    user_message = build_user_message(doc_params)
    llm_output = call_llm(
        user_message,
        doc_params,
        report,
        run_id,
        cached_input=not first_report,
        model=args.model,
    )

    result_payload = build_result_payload(
        llm_output["result"],
        report,
        llm_output["cost_log"],
        extraction_target,
        extraction_meta=report_output.get("extraction"),
        warnings=report_output["warnings"],
    )
    result_target = result_path(output_dirs, report)
    result_target.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    archived_target = archived_result_path(run_output_dirs, report)
    archived_target.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    append_jsonl(first_run_log_path(output_dirs), llm_output["cost_log"])
    append_jsonl(archived_first_run_log_path(run_output_dirs), llm_output["cost_log"])
    report_output["first_run_log"] = llm_output["cost_log"]
    report_output["result_path"] = str(result_target)
    report_output["archived_result_path"] = str(archived_target)
    report_output["success"] = True

    if args.consistency_runs > 0:
        consistency = consistency_check(
            user_message,
            doc_params,
            report,
            run_id,
            n_runs=args.consistency_runs,
            model=args.model,
        )
        for cost_log in consistency["cost_logs"]:
            append_jsonl(consistency_log_path(output_dirs), cost_log)
            append_jsonl(archived_consistency_log_path(run_output_dirs), cost_log)
        report_output["consistency"] = consistency["summary"]

    return report_output


def load_reports(args: argparse.Namespace) -> list[ReportSpec]:
    if not args.manifest.is_absolute():
        args.manifest = (BASE_DIR / args.manifest).resolve()
    reports = [
        report
        for report in load_manifest(args.manifest)
        if report.issuer.lower() == args.issuer.lower()
    ]
    if args.limit > 0:
        reports = reports[: args.limit]
    return reports


def main() -> int:
    args = parse_args()
    reports = load_reports(args)
    if not reports:
        print(f"No reports found for issuer '{args.issuer}' in {args.manifest}")
        return 1

    output_dirs = ensure_output_dirs(args.issuer)
    run_id = utc_run_id()
    run_output_dirs = ensure_run_output_dirs(output_dirs, run_id)

    print("=" * 72)
    print(f"LLM Report Batch Runner - issuer={args.issuer} - run_id={run_id}")
    print("=" * 72)
    print(f"Manifest: {args.manifest}")
    print(f"Reports queued: {len(reports)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Skip extract: {args.skip_extract}")
    print(f"Consistency runs: {args.consistency_runs}")

    first_run_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    failures = 0

    for index, report in enumerate(reports, start=1):
        print(f"\n[{index}/{len(reports)}] {report.document_id} - {report.source_pdf.name}")
        try:
            outcome = process_report(
                report,
                output_dirs,
                run_output_dirs,
                run_id,
                args,
                first_report=(index == 1),
            )
            for warning in outcome["warnings"]:
                print(f"  Warning: {warning}")

            if outcome["dry_run"]:
                print(f"  Extracted text ready at {extracted_text_path(output_dirs, report)}")
                print(f"  Planned result path: {outcome['planned_result_path']}")
                continue

            first_run_rows.append(outcome["first_run_log"])
            print(
                "  First run: "
                f"{outcome['first_run_log']['signal']} "
                f"(cost ${outcome['first_run_log']['estimated_cost_usd']:.5f})"
            )
            if outcome["consistency"] is not None:
                consistency_rows.append(outcome["consistency"])
                print(
                    "  Consistency: "
                    f"agreement={outcome['consistency']['signal_agreement']} "
                    f"spread={outcome['consistency']['score_range']}"
                )
        except Exception as exc:
            failures += 1
            failure_log = {
                "run_id": run_id,
                "status": "failed",
                "document_id": report.document_id,
                "issuer": report.issuer,
                "company": report.company,
                "ticker": report.ticker,
                "fiscal_period": report.fiscal_period,
                "report_date": report.report_date,
                "source_pdf": str(report.source_pdf),
                "error": str(exc),
            }
            append_jsonl(first_run_log_path(output_dirs), failure_log)
            print(f"  Failed: {exc}")

    if first_run_rows:
        summary_target = first_run_summary_path(output_dirs, args.issuer)
        write_summary_csv(summary_target, first_run_rows)
        archived_summary_target = archived_first_run_summary_path(
            run_output_dirs, args.issuer, run_id
        )
        write_summary_csv(archived_summary_target, first_run_rows)
        print(f"\nFirst-run summary written to {summary_target}")
        print(f"Run-specific first-run summary written to {archived_summary_target}")

    if consistency_rows:
        summary_payload = {"run_id": run_id, "issuer": args.issuer, "reports": consistency_rows}
        consistency_target = consistency_summary_path(output_dirs, args.issuer)
        write_json(consistency_target, summary_payload)
        archived_consistency_target = archived_consistency_summary_path(
            run_output_dirs, args.issuer, run_id
        )
        write_json(archived_consistency_target, summary_payload)
        print(f"Consistency summary written to {consistency_target}")
        print(f"Run-specific consistency summary written to {archived_consistency_target}")

    batch_metadata = {
        "run_id": run_id,
        "issuer": args.issuer,
        "manifest": str(args.manifest),
        "dry_run": args.dry_run,
        "skip_extract": args.skip_extract,
        "consistency_runs": args.consistency_runs,
        "model": args.model,
        "hold_upper": args.hold_upper,
        "hold_lower": args.hold_lower,
        "reports_queued": [report.document_id for report in reports],
        "failures": failures,
        "first_run_summary_path": (
            str(first_run_summary_path(output_dirs, args.issuer)) if first_run_rows else ""
        ),
        "consistency_summary_path": (
            str(consistency_summary_path(output_dirs, args.issuer)) if consistency_rows else ""
        ),
    }
    write_json(run_output_dirs["root"] / "batch_metadata.json", batch_metadata)

    if args.dry_run:
        print("\nDry run complete.")
        return 0

    if failures:
        print(f"\nCompleted with {failures} failed report(s).")
        return 1

    total_cost = sum(row["estimated_cost_usd"] for row in first_run_rows)
    print(f"\nCompleted successfully. First-run total cost: ${total_cost:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
