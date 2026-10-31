"""CPU coverage for accuracy-before-profile ordering and accepted performance exports."""

import csv
import json

import pytest

from . import run
from .framework.cases import CaseSpec, save_cases
from .profiling.msprof_parser import parse_benchmark


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def accuracy_reports(directory, *, fail_second=False):
    for rank in range(2):
        write_json(
            directory / f"rank-{rank}.json",
            {
                "global_rank": rank,
                "passed": True,
                "accuracy_checked": True,
                "accuracy": {"passed": True},
                "changed_data": {"passed": not (fail_second and rank == 1)},
            },
        )


@pytest.mark.parametrize("failure", [None, "second_check", "launcher_exit"])
def test_perf_accuracy_gate_precedes_profiler(tmp_path, monkeypatch, failure):
    phases = []

    def execute(command, log, timeout, env):
        phases.append(command[0])
        if command[0] == "accuracy":
            accuracy_reports(tmp_path / "accuracy", fail_second=failure == "second_check")
            if failure == "launcher_exit":
                raise RuntimeError("worker failed after writing rank reports")

    monkeypatch.setattr(run, "execute", execute)
    arguments = dict(mode="perf", node_rank=0, world_size=2, timeout=10, env={}, accuracy_command=["accuracy"])
    if failure:
        with pytest.raises(RuntimeError):
            run.execute_case(tmp_path, ["bench"], **arguments)
    else:
        run.execute_case(tmp_path, ["bench"], **arguments)
    assert phases == (["accuracy"] if failure else ["accuracy", "bench"])
    completion = json.loads((tmp_path / "launcher-node-0.json").read_text())
    assert completion["completed"] is (failure is None)


@pytest.mark.parametrize("complete,damaged", [(False, False), (True, False), (True, True)])
def test_report_requires_successful_launcher_and_uses_iteration_max(tmp_path, complete, damaged):
    case = CaseSpec()
    directory = tmp_path / case.case_id
    directory.mkdir()
    save_cases(tmp_path / "cases.json", [case])
    write_json(
        tmp_path / "plan.json",
        {
            "mode": "perf",
            "require_launcher_completion": True,
            "arguments": {"nnodes": 1, "peak_tflops": 378},
            "selections": [{"case_id": case.case_id, "world_size": 2}],
        },
    )
    accuracy_reports(directory / "accuracy")
    # Median(max across ranks) is 100, whereas max(median for each rank) is 51.
    for rank, samples in enumerate(([1, 100], [100, 2])):
        write_json(directory / f"rank-{rank}.json", {"global_rank": rank, "passed": True})
        write_json(
            directory / f"capture-{rank}.json",
            {
                "global_rank": rank,
                "passed": True,
                "metric": "msprof_task_duration_us",
                "samples_us": samples,
            },
        )
    if complete:
        write_json(directory / "launcher-node-0.json", {"node_rank": 0, "completed": True})
    if damaged:
        (directory / "capture-1.json").write_text('{"global_rank":')
    report = run.report_run(tmp_path)
    accepted = complete and not damaged
    assert report["passed"] is accepted
    with (tmp_path / "performance.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    if accepted:
        assert float(row["kernel_us_per_call"]) == 100
        flops = 2 * 2 * case.m * case.k * case.n
        assert float(row["mfu_percent"]) == pytest.approx(100 * flops / (100 * 1e6 * 2 * 378))
    else:
        assert row["kernel_us_per_call"] == row["mfu_percent"] == ""
    assert "event_us_per_call" not in row


@pytest.mark.parametrize("mode", ["bench", "perf"])
def test_late_peer_cannot_erase_launcher_timeout(tmp_path, monkeypatch, mode):
    phases = []

    def execute(command, log, timeout, env):
        phases.append(command[0])
        if command[0] == "accuracy":
            accuracy_reports(tmp_path / "accuracy")

    monkeypatch.setattr(run, "execute", execute)
    with pytest.raises(TimeoutError):
        run.execute_case(
            tmp_path,
            ["bench"],
            mode=mode,
            node_rank=0,
            world_size=2,
            timeout=0,
            env={},
            accuracy_command=["accuracy"],
            nnodes=2,
        )
    write_json(tmp_path / "launcher-node-1.json", {"node_rank": 1, "completed": True})
    assert json.loads((tmp_path / "launcher-node-0.json").read_text())["completed"] is False
    assert phases == (["accuracy"] if mode == "perf" else ["bench"])
    if mode == "perf":
        assert json.loads((tmp_path / "accuracy/launcher-node-0.json").read_text())["completed"] is False


def test_capture_ignores_other_kernels_and_rejects_extra_target_launches(tmp_path):
    path = tmp_path / "op_summary_0.csv"
    fields = ["Op Name", "Device_id", "Task Start Time(us)", "Task Duration(us)"]
    rows = [["Ascend950AllToAllVGroupedMatmul", "0", i * 100, value] for i, value in enumerate((999, 10, 20))]
    rows.append(["ordinary_fp32_matmul", "0", 50, 10000])

    def write_rows():
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    write_rows()
    capture = parse_benchmark(tmp_path, "alltoallv_gmm", warmup=1, iterations=2, batches=1)
    assert capture["passed"] and capture["samples_us"] == [10, 20]
    rows.append(["Ascend950AllToAllVGroupedMatmul", "0", 400, 30])
    write_rows()
    assert not parse_benchmark(tmp_path, "alltoallv_gmm", warmup=1, iterations=2, batches=1)["passed"]
