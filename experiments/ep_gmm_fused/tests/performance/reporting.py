"""Accuracy gates and labelled kernel-performance exports shared by EP sizes."""

import csv
import json
from pathlib import Path


def read_reports(directory, pattern):
    reports, errors = [], []
    for path in sorted(Path(directory).glob(pattern)):
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict):
                raise ValueError("report must be a JSON object")
            reports.append(report)
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    return reports, errors


def accuracy_gate(directory, world_size):
    reports, errors = read_reports(directory, "rank-*.json")
    ranks = [item.get("global_rank") for item in reports]
    if len(ranks) != world_size or set(ranks) != set(range(world_size)):
        errors.append("missing or duplicate accuracy rank reports")
    for item in reports:
        if not (
            item.get("passed") is True
            and item.get("accuracy_checked") is True
            and item.get("accuracy", {}).get("passed") is True
            and item.get("changed_data", {}).get("passed") is True
        ):
            errors.append(f"rank {item.get('global_rank')} did not pass both full checks")
    return {"passed": not errors, "errors": errors, "rank_reports": reports}


def case_display(case):
    scenario = case.scenario or {}
    top_k = scenario.get("top_k")
    tokens = scenario.get("global_tokens", case.m // top_k * case.ep_size) if top_k else None
    suffix = f"globalT{tokens}_TopK{top_k}" if top_k else f"localM{case.m}"
    return {
        "case_name": f"{case.operator}_ep{case.ep_size}_G{case.local_experts}_K{case.k}_N{case.n}_{suffix}_route-{case.route}",
        "case_parameters": {
            "operator": case.operator,
            "ep_size": case.ep_size,
            "g": case.local_experts,
            "k": case.k,
            "n": case.n,
            "global_tokens": tokens,
            "top_k": top_k,
            "expanded_rows_per_rank": case.m,
            "route": case.route,
        },
    }


def write_performance_csv(root, results, cases, peak_tflops=None):
    """Only accepted complete captures have metrics; raw evidence is retained separately."""
    fields = [
        "case_id",
        "case_name",
        "operator",
        "ep_size",
        "g",
        "k",
        "n",
        "global_tokens",
        "top_k",
        "expanded_rows_per_rank",
        "route",
        "status",
        "accuracy_status",
        "timing_backend",
        "kernel_us_per_call",
        "kernel_p90_us",
        "kernel_p99_us",
        "useful_gemm_flops_global",
        "global_tflops",
        "peak_tflops_per_device",
        "mfu_percent",
    ]
    rows = []
    with (Path(root) / "performance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            case = cases[result["case_id"]]
            display = case_display(case)
            row = {
                "case_id": case.case_id,
                "case_name": display["case_name"],
                **display["case_parameters"],
                "status": result["status"],
                "timing_backend": "msprof_op_summary",
                "accuracy_status": "passed"
                if result.get("accuracy", {}).get("passed")
                else "failed_or_incomplete"
                if "accuracy" in result
                else "not_run",
            }
            if result["passed"] and "msprof" in result:
                timing = result["msprof"]
                flops = 2 * case.ep_size * case.m * case.k * case.n * (2 if case.operator == "composition" else 1)
                throughput = flops / (timing["p50_us"] * 1e6)
                row.update(
                    kernel_us_per_call=timing["p50_us"],
                    kernel_p90_us=timing["p90_us"],
                    kernel_p99_us=timing["p99_us"],
                    useful_gemm_flops_global=flops,
                    global_tflops=throughput,
                )
                if peak_tflops is not None:
                    row.update(
                        peak_tflops_per_device=peak_tflops, mfu_percent=100 * throughput / (case.ep_size * peak_tflops)
                    )
            writer.writerow(row)
            rows.append(row)
    return rows
