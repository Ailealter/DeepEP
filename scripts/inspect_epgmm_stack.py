#!/usr/bin/env python3
"""Read installed EP_GMM_FUSED kernel stack metadata without loading any .so.

Run from this repository with the Python environment containing ascend-deepep:
  python scripts/inspect_epgmm_stack.py

Optional: --binary /path/to/kernel.o (repeatable), --json /path/to/report.json.
--max-stack-bytes 32768 checks the GM scheduler build's selected stack budget.
--acl-config /path/to/new.json writes a separate TorchNPU initialization config
with a 32 KiB stack by default, preserving the other ACL configuration fields.
Only the standard library is used. No installation, compilation or NPU calls.
The reader scans ordinary embedded ELF images; compressed/unknown formats are
reported as incomplete, never as zero stack usage or a successful device test.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys


class InvalidElf(ValueError):
    pass


def sections(data, base):
    """Return file-backed sections of one ELF image, with bounded offsets."""
    ident = data[base : base + 16]
    if len(ident) != 16 or ident[:4] != b"\x7fELF" or ident[6] != 1:
        raise InvalidElf("bad ELF identification")
    if ident[4] not in (1, 2) or ident[5] not in (1, 2):
        raise InvalidElf("unsupported ELF class or byte order")
    endian = "<" if ident[5] == 1 else ">"
    header = struct.Struct(endian + ("16sHHIQQQIHHHHHH" if ident[4] == 2 else "16sHHIIIIIHHHHHH"))
    entry = struct.Struct(endian + ("IIQQQQIIQQ" if ident[4] == 2 else "IIIIIIIIII"))
    if base + header.size > len(data):
        raise InvalidElf("truncated ELF header")
    h = header.unpack_from(data, base)
    shoff, shentsize, count, names_index = h[6], h[11], h[12], h[13]
    if h[3] != 1 or h[8] < header.size or shentsize < entry.size or not shoff:
        raise InvalidElf("missing/invalid section table")
    table = base + shoff
    if table + shentsize > len(data):
        raise InvalidElf("section table out of bounds")
    first = entry.unpack_from(data, table)
    if count == 0:
        count = first[5]
    if names_index == 0xFFFF:
        names_index = first[6]
    if not 0 < count <= 100000 or not 0 < names_index < count:
        raise InvalidElf("invalid section counts")
    if table + count * shentsize > len(data):
        raise InvalidElf("truncated section table")
    entries = [entry.unpack_from(data, table + i * shentsize) for i in range(count)]
    names = entries[names_index]
    start, end = base + names[4], base + names[4] + names[5]
    if names[1] != 3 or start > len(data) or end > len(data):
        raise InvalidElf("invalid section-name table")
    strings = data[start:end]
    result = []
    for s in entries:
        if s[0] >= len(strings):
            raise InvalidElf("invalid section name offset")
        nul = strings.find(b"\0", s[0])
        if nul == -1:
            raise InvalidElf("unterminated section name")
        name = strings[s[0] : nul].decode("ascii", errors="replace")
        if s[1] in (0, 8):  # NULL and NOBITS have no file-backed contents.
            continue
        start, end = base + s[4], base + s[4] + s[5]
        if start > len(data) or end > len(data):
            raise InvalidElf("section data out of bounds")
        result.append((name, s[2], data[start:end]))
    return endian, result


def min_stack(payload, endian):
    """Ascend per-kernel TLV: uint16 type, uint16 value length, value bytes."""
    offset, values = 0, []
    while offset < len(payload):
        if len(payload) - offset < 4:
            raise InvalidElf("truncated TLV header")
        tag, size = struct.unpack_from(endian + "HH", payload, offset)
        offset += 4
        if offset + size > len(payload):
            raise InvalidElf("truncated TLV value")
        if tag == 8:  # RT_FUNCTION_TYPE_SU_STACK_SIZE
            if size != 4:
                raise InvalidElf("unexpected stack TLV length")
            values.append(struct.unpack_from(endian + "I", payload, offset)[0])
        offset += size
    if not values:
        raise InvalidElf("stack TLV type 8 not present")
    return max(values)


def family(name):
    if "Ascend950AllToAllVGroupedMatmul" in name:
        return "alltoallv_gmm"
    if "GroupedMatmulAllToAllVTla" in name:
        return "gmm_alltoallv"
    return "other"


def inspect(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    records, problems, valid = [], [], 0
    offset = data.find(b"\x7fELF")
    while offset >= 0:
        try:
            endian, contents = sections(data, offset)
        except (InvalidElf, struct.error):
            pass  # An ELF magic sequence in data is not necessarily an image.
        else:
            valid += 1
            for name, flags, payload in contents:
                if not name.startswith(".ascend.meta."):
                    continue
                kernel = name[len(".ascend.meta.") :]
                try:
                    if flags & 0x800:  # SHF_COMPRESSED
                        raise InvalidElf("compressed metadata")
                    size = min_stack(payload, endian)
                except InvalidElf as error:
                    problems.append(dict(elf_offset=offset, kernel=kernel, error=str(error)))
                else:
                    records.append(dict(elf_offset=offset, kernel=kernel, family=family(kernel), min_stack_bytes=size))
        offset = data.find(b"\x7fELF", offset + 4)
    return dict(
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        valid_elf_images=valid,
        records=records,
        unreadable_metadata=problems,
    )


def installed_binary():
    dist = importlib.metadata.distribution("ascend-deepep")
    found = [
        dist.locate_file(p)
        for p in (dist.files or [])
        if str(p).replace("\\", "/").startswith("deep_ep_experimental/ep_gmm_fused/_C") and str(p).endswith(".so")
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one installed EP_GMM_FUSED _C.so, found {len(found)}; use --binary")
    return found[0]


def checked_stack_size(reports, limit=32768):
    """Fail closed on missing/partial metadata, including non-GMM kernels."""
    for report in reports:
        if report["unreadable_metadata"]:
            raise ValueError("cannot validate a module with unreadable kernel stack metadata")
        for name in ("alltoallv_gmm", "gmm_alltoallv"):
            if not any(row["family"] == name for row in report["records"]):
                raise ValueError(f"missing {name} stack metadata")
    largest = max((row["min_stack_bytes"] for report in reports for row in report["records"]), default=0)
    if not reports or largest == 0:
        raise ValueError("missing nonzero kernel stack requirements")
    if largest > limit:
        raise ValueError(f"kernel requires {largest} stack bytes, exceeds configured ceiling {limit}")
    return largest


def default_acl_config():
    # Locate package data without importing torch/torch_npu or initializing NPU.
    dist = importlib.metadata.distribution("torch-npu")
    path = Path(dist.locate_file("torch_npu/acl_default.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ACL default config must be an object: {path}")
    return path, value


def write_acl_config(output, reports, source=None, runtime_stack_bytes=32768):
    """Create a per-run config; never overwrite package data or user files."""
    if runtime_stack_bytes not in range(32768, 131073, 16384):
        raise ValueError("runtime stack must be a multiple of 16 KiB between 32 and 128 KiB")
    required = checked_stack_size(reports, runtime_stack_bytes)
    default_path, defaults = default_acl_config()
    source = source or os.environ.get("TORCH_ACL_INIT_CONFIG_PATH")
    source_path = Path(source).resolve() if source else default_path
    config = json.loads(source_path.read_text(encoding="utf-8")) if source else defaults.copy()
    if not isinstance(config, dict):
        raise ValueError("ACL config must be a JSON object")
    # TorchNPU's CANN9.2 lazy-device path relies on this default. Preserve the
    # selected device when supplied, and otherwise inherit the installed default.
    if "defaultDevice" not in config:
        default_device = defaults.get("defaultDevice")
        if not isinstance(default_device, dict):
            raise ValueError("installed TorchNPU default config has no valid defaultDevice")
        config["defaultDevice"] = default_device.copy()
    device = config["defaultDevice"]
    if not isinstance(device, dict) or "default_device" not in device:
        raise ValueError("ACL config must preserve TorchNPU defaultDevice.default_device")
    stack = config.setdefault("StackSize", {})
    if not isinstance(stack, dict):
        raise ValueError("ACL StackSize must be an object")
    old = stack.get("aicore_stack_size", 32768)
    if isinstance(old, bool) or not isinstance(old, int) or not 0 <= old <= 131072:
        raise ValueError("existing aicore_stack_size is not valid for Ascend950 (maximum 131072)")
    # The GM scheduler build defaults to 32 KiB. Other fields in a user's config survive;
    # the selected stack budget is explicit and never bypasses the ELF check.
    stack["aicore_stack_size"] = runtime_stack_bytes
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("ACL_CONFIG_SOURCE:", source_path)
    print("ACL_CONFIG:", Path(output).resolve())
    print(f"ACL_STACK_BYTES: {runtime_stack_bytes} (ELF maximum required: {required})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", action="append", type=Path)
    parser.add_argument("--json", type=Path, help="optional full metadata report; will not overwrite a file")
    parser.add_argument("--max-stack-bytes", type=int, help="fail if any kernel exceeds this byte count")
    parser.add_argument("--acl-config", type=Path, help="create a per-run ACL config for Ascend950; never overwrite")
    parser.add_argument(
        "--acl-source", type=Path, help="base ACL JSON; otherwise use TORCH_ACL_INIT_CONFIG_PATH/default"
    )
    parser.add_argument(
        "--runtime-stack-bytes",
        type=int,
        default=32768,
        help="ACL stack budget, default 32768 for the GM scheduler build",
    )
    args = parser.parse_args()
    if args.max_stack_bytes is not None and args.max_stack_bytes <= 0:
        parser.error("--max-stack-bytes must be positive")
    if args.acl_source and not args.acl_config:
        parser.error("--acl-source requires --acl-config")
    reports = [inspect(p) for p in (args.binary or [installed_binary()])]
    incomplete = False
    for r in reports:
        print("BINARY:", r["path"])
        print("SHA256:", r["sha256"])
        print("ELF_IMAGES:", r["valid_elf_images"], "STACK_RECORDS:", len(r["records"]))
        for name in ("alltoallv_gmm", "gmm_alltoallv"):
            rows = [x for x in r["records"] if x["family"] == name]
            if not rows:
                print(name + ": UNKNOWN (no readable stack metadata)")
                incomplete = True
                continue
            size = max(x["min_stack_bytes"] for x in rows)
            print(
                f"{name}: max_min_stack={size} bytes ({size / 1024:.2f} KiB), records={len(rows)}, "
                f"over_32KiB={size > 32768}, over_128KiB={size > 131072}"
            )
        if r["unreadable_metadata"]:
            print("UNREADABLE_METADATA:", len(r["unreadable_metadata"]), "(partial results; use --json for details)")
            incomplete = True
    if args.json:
        with args.json.open("x", encoding="utf-8") as stream:
            json.dump(reports, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("REPORT:", args.json.resolve())
    if args.max_stack_bytes is not None:
        size = checked_stack_size(reports, args.max_stack_bytes)
        print(f"STACK_GATE: passed, max={size}, ceiling={args.max_stack_bytes}")
    if args.acl_config:
        write_acl_config(args.acl_config, reports, args.acl_source, args.runtime_stack_bytes)
    print("NO_NPU: no native module was loaded; no build/install/NPU operation.")
    if incomplete:
        print("INCOMPLETE: missing metadata is not zero usage; inspect the uncompressed device ELF/build objects.")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        print("INSPECTION_FAILED:", error, file=sys.stderr)
        raise SystemExit(2)
