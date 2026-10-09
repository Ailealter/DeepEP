"""Notify per-AIV diagnostics and automatic SVG / Perfetto export (CPU only)."""

import csv
import html
import json
from pathlib import Path
import statistics

MAGIC = 0x4E544659
VERSION = 1
WORDS = 64
HEADER_WORDS = 16
TICKS_PER_US = 1000  # Ascend950 GetSystemCycle system counter, as in Combine.
PHASES = (
    "Setup",
    "ParseTopk",
    "LocalWaitParseTopk",
    "LocalPrefixSegments",
    "LocalWaitLocalPrefixSegments",
    "LocalPrefixTotals",
    "LocalWaitLocalPrefixTotals",
    "FinalizeTopk",
    "PackCounts",
    "LocalWaitFinalizeTopk",
    "CountExchange",
    "GlobalWaitCountExchange",
    "ExpertTotals",
    "LocalWaitExpertTotals",
    "RankBases",
    "LocalWaitRankBases",
    "AddressStride",
    "LocalWaitAddressStride",
    "ResolveDst",
    "ForwardCount",
    "LocalWaitResolveDst",
    "ForwardPrefix",
    "LocalWaitForwardPrefix",
    "ForwardWrite",
    "LocalWaitForwardWrite",
    "ForwardExchange",
    "GlobalWaitForwardExchange",
    "GatherCount",
    "LocalWaitGatherCount",
    "GatherSourcePrefix",
    "LocalWaitGatherSourcePrefix",
    "GatherPrefix",
    "LocalWaitGatherPrefix",
    "BuildForwardRows",
    "ChunkRangesBackwardCount",
    "LocalWaitBuildForwardRows",
    "BackwardChunkPrefix",
    "LocalWaitBackwardChunkPrefix",
    "BackwardTaskPrefix",
    "LocalWaitBackwardTaskPrefix",
    "BackwardWrite",
    "LocalWaitBackwardWrite",
    "BackwardGatherExchange",
    "GlobalWaitBackwardGatherExchange",
    "CollectBackward",
)
COLORS = dict(compute="#478ac6", exchange="#32a58b", local_wait="#efa645", global_wait="#d16d85")
NOTES = [
    "Instrumented invocation; use the separate notify_full_kernel profiler samples for performance.",
    "Diagnostic-only entry barrier reduces launch skew. Time zero is its earliest AIV exit on this rank; device clocks are not calibrated.",
    "Work stages include their existing VF/MTE drains. Wait stages measure existing barriers, not pure wire latency.",
    "FullNotify only: excludes final profile export, host-launched post-Notify barrier, peer-plan kernel and totals copy.",
]


def category(name):
    if name.startswith("LocalWait"):
        return "local_wait"
    if name.startswith("GlobalWait"):
        return "global_wait"
    return "exchange" if name.endswith("Exchange") else "compute"


def decode(rows, *, rank, world, tokens, topk, experts, chunk_tokens, sample=1):
    if len(rows) not in (32, 64):
        raise ValueError("Notify profile requires 32/64 AIV rows")
    if world not in (2, 4, 8, 16, 32, 64, 128, 256, 512) or not 0 <= rank < world:
        raise ValueError("Invalid Notify profile topology")
    if not (1 <= tokens <= 16384 and 1 <= topk <= 16 and world <= experts <= 1024 and experts % world == 0):
        raise ValueError("Invalid Notify profile shape")
    if chunk_tokens not in (256, 512, 1024) or type(sample) is not int or not 0 < sample < 2**63:
        raise ValueError("Invalid Notify chunk size or sample")
    stamps = []
    for core, row in enumerate(rows):
        if len(row) != WORDS or any(type(v) is not int or not 0 <= v < 2**63 for v in row):
            raise ValueError(f"Invalid Notify profile row {core}")
        expected = [
            MAGIC,
            VERSION,
            rank,
            core,
            world,
            tokens,
            topk,
            experts,
            chunk_tokens,
            sample,
            len(rows),
            len(PHASES),
        ]
        if row[:12] != expected or row[12] > 63 or row[15] != 1:
            raise ValueError(f"Notify profile header mismatch on AIV {core}; rebuild wheel or check stale sample")
        times = row[HEADER_WORDS : HEADER_WORDS + len(PHASES) + 1]
        if not 0 < row[13] <= row[14] <= times[0]:
            raise ValueError(f"Invalid Notify entry barrier timestamps on AIV {core}")
        if not times[0] or times[-1] <= times[0] or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f"Missing/non-monotonic Notify timestamps on AIV {core}")
        if any(row[HEADER_WORDS + len(PHASES) + 1 :]):
            raise ValueError("Unexpected Notify profile trailing data")
        stamps.append(times)
    origin = min(row[14] for row in rows)
    events = [
        dict(
            core=core,
            name=name,
            category=category(name),
            begin_us=(times[i] - origin) / TICKS_PER_US,
            end_us=(times[i + 1] - origin) / TICKS_PER_US,
            duration_us=(times[i + 1] - times[i]) / TICKS_PER_US,
        )
        for core, times in enumerate(stamps)
        for i, name in enumerate(PHASES)
    ]
    phases = []
    for name in PHASES:
        values = [e for e in events if e["name"] == name]
        slowest = max(values, key=lambda e: e["duration_us"])
        phases.append(
            dict(
                name=name,
                category=category(name),
                mean_us=statistics.mean(e["duration_us"] for e in values),
                max_us=slowest["duration_us"],
                max_core=slowest["core"],
            )
        )
    return dict(
        version=VERSION,
        operator="notify",
        kernel="notify_profile_kernel",
        rank=rank,
        world_size=world,
        tokens=tokens,
        topk=topk,
        experts=experts,
        chunk_tokens=chunk_tokens,
        sample=sample,
        ticks_per_us=TICKS_PER_US,
        origin_ticks=origin,
        num_cores=len(rows),
        cache_flags=dict(
            zip(
                (
                    "parse_hist",
                    "parse_routes",
                    "forward_inputs",
                    "local_prefix",
                    "local_prefix_totals",
                    "forward_counts",
                ),
                (bool(rows[0][12] & (1 << bit)) for bit in range(6)),
            )
        ),
        alignment=dict(
            kind="diagnostic_entry_global_barrier",
            calibrated_clocks=False,
            wait_us_by_core=[(row[14] - row[13]) / TICKS_PER_US for row in rows],
            exit_skew_us=(max(row[14] for row in rows) - origin) / TICKS_PER_US,
        ),
        raw=rows,
        events=events,
        phases=phases,
        span_us=(max(t[-1] for t in stamps) - origin) / TICKS_PER_US,
        notes=NOTES,
    )


def perfetto_trace(data):
    rank = data["rank"]
    events = [dict(ph="M", name="process_name", pid=rank, args=dict(name=f"Notify rank {rank}"))]
    for core in range(data["num_cores"]):
        events.append(dict(ph="M", name="thread_name", pid=rank, tid=core, args=dict(name=f"AIV {core}")))
    for e in data["events"]:
        events.append(
            dict(
                ph="X",
                name=e["name"],
                cat=e["category"],
                pid=rank,
                tid=e["core"],
                ts=e["begin_us"],
                dur=e["duration_us"],
                args=dict(sample=data["sample"]),
            )
        )
    return dict(traceEvents=events, displayTimeUnit="ns", metadata=dict(notes=data["notes"]))


def write_timeline(data, destination):
    """Write raw+decoded JSON and ready-to-open views beside it. Return paths."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = dict(
        json=str(destination),
        svg=str(destination.with_suffix(".svg")),
        perfetto=str(destination.with_suffix(".perfetto.json")),
        csv=str(destination.with_suffix(".csv")),
    )
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    Path(paths["perfetto"]).write_text(json.dumps(perfetto_trace(data)) + "\n", encoding="utf-8")
    with Path(paths["csv"]).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("core", "name", "category", "begin_us", "end_us", "duration_us"))
        writer.writeheader()
        writer.writerows(data["events"])
    width, left, plot, top, stride = 2400, 95, 2260, 155, 28
    height = top + data["num_cores"] * stride + 35
    scale = plot / data["span_us"]
    title = f"Notify EP{data['world_size']} / T{data['tokens']} / chunk{data['chunk_tokens']} / rank {data['rank']}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font:12px monospace;fill:#233047} .title{font:bold 19px sans-serif}</style>",
        f'<text class="title" x="20" y="30">{html.escape(title)}</text>',
        '<text x="20" y="53">Separate diagnostic invocation; hover blocks for stage / start / end / duration.</text>',
        '<text x="20" y="73">Zero: local entry-barrier exit (clocks not calibrated). Entry wait, profile export and trailing host kernels excluded.</text>',
    ]
    for i, (name, color) in enumerate(COLORS.items()):
        x = 20 + i * 220
        lines += [
            f'<rect x="{x}" y="87" width="14" height="14" fill="{color}"/>',
            f'<text x="{x + 21}" y="99">{name}</text>',
        ]
    for i in range(11):
        x = left + i * plot / 10
        lines += [
            f'<line x1="{x}" y1="135" x2="{x}" y2="{height - 20}" stroke="#e5e7eb"/>',
            f'<text x="{x}" y="126" text-anchor="middle">{i * data["span_us"] / 10:.1f} us</text>',
        ]
    for core in range(data["num_cores"]):
        lines.append(f'<text x="15" y="{top + core * stride + 15}">AIV {core}</text>')
    for e in data["events"]:
        x, y, w = left + e["begin_us"] * scale, top + e["core"] * stride, e["duration_us"] * scale
        tip = f"AIV {e['core']} {e['name']}: {e['begin_us']:.3f} .. {e['end_us']:.3f} us; duration {e['duration_us']:.3f} us"
        lines.append(
            f'<rect x="{x:.3f}" y="{y}" width="{w:.3f}" height="21" fill="{COLORS[e["category"]]}"><title>{html.escape(tip)}</title></rect>'
        )
        if w > len(e["name"]) * 7 + 4:
            lines.append(f'<text x="{x + 3:.3f}" y="{y + 15}">{html.escape(e["name"])}</text>')
    lines.append("</svg>")
    Path(paths["svg"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
