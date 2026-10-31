#!/usr/bin/env bash
# 仓内 EP_GMM 入口：git pull 后运行 bash scripts/run_ep16_gmm.sh <命令>。
# build 编译安装；accuracy 做两轮全输出精度；perf 通过两轮精度后独立采集 msprof。
# 默认 accuracy 和 perf 均为 ep8-baseline：两个固定 EP8 回归例。
# EP8 通过后显式选择 EP16 smoke/all；all 精度1152例，性能仅前两个算子768例。
# 性能采用 op_summary 目标 kernel duration，不采集 composition 串联用例。
# 每次调用先跨 rank 取最大耗时，再计算 p50；融合 kernel 时长用于 MFU。
# plan / plan-perf 仅生成计划，不加载 NPU 扩展或执行用例。
# build/accuracy/perf 加载扩展前检查 EP_GMM ELF，并使用32 KiB ACL配置。
# build 保留 MegaMoE 和已有 timer；不自动准备 SHMEM/CatCCOS/CATLASS 依赖。
# 测试模块和检查器跟随本仓库版本，不使用旧 EPGMM_TEST_SCRIPT / EPGMM_STACK_INSPECTOR。

# ======================= 常用配置：主要修改这里 =======================
# 默认源码和依赖来自本脚本所在仓库；显式变量可覆盖源码/依赖目录。
epgmm_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EPGMM_REPO="${EPGMM_REPO:-$(cd -- "$epgmm_script_dir/.." && pwd)}"
epgmm_stack_inspector="$epgmm_script_dir/inspect_epgmm_stack.py"
EPGMM_DEPS_REPO="${EPGMM_DEPS_REPO:-$EPGMM_REPO}"

# 已激活 py311_lzy 时直接使用；否则通过 conda.sh 激活。
# 如果 Conda 不在 /root/miniconda3，请修改 CONDA_SH；留空 CONDA_ENV 可用当前 Python。
EPGMM_CONDA_ENV="${EPGMM_CONDA_ENV-py311_lzy}"
EPGMM_CONDA_SH="${EPGMM_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
EPGMM_PYTHON="${EPGMM_PYTHON:-python}"

# 已加载 CANN 时留空；希望脚本加载时，填写你实际使用的 set_env.sh 完整路径。
EPGMM_CANN_ENV="${EPGMM_CANN_ENV:-}"

# 套件用空格分隔，按顺序执行：
# ep8-baseline = EP8 两例；baseline = EP2 精度3例、性能2例。
# EP16 精度：smoke 30例、topk 450例、scale 672例、all 1152例。
# EP16 性能：smoke 20例、topk 300例、scale 448例、all 768例，排除 composition。
# all = smoke + topk + scale；不要再与这三个套件重复组合。
# 先保持默认完成 EP8 回归，再显式选择 "smoke" 或 "all" 测试 EP16。
EPGMM_SUITES="${EPGMM_SUITES:-ep8-baseline}"
# perf / plan-perf 独立选择套件，同样默认先测 EP8。
EPGMM_PERF_SUITES="${EPGMM_PERF_SUITES:-ep8-baseline}"
# 每例完成精度检查后，先预热10次，再测5批、每批20次API调用。
EPGMM_WARMUP="${EPGMM_WARMUP:-10}"
EPGMM_ITERATIONS="${EPGMM_ITERATIONS:-20}"
EPGMM_BATCHES="${EPGMM_BATCHES:-5}"
# 当前机器为 Ascend 950PR、28 Cube，单设备 BF16 理论峰值按378 TFLOPS计算。
# 换其他 SKU 时修改此值；不要用实测 GEMM 吞吐替代理论峰值来命名 MFU。
EPGMM_PEAK_TFLOPS="${EPGMM_PEAK_TFLOPS:-378}"
EPGMM_BASELINE_DEVICES="${EPGMM_BASELINE_DEVICES:-0,1}"
EPGMM_EP8_DEVICES="${EPGMM_EP8_DEVICES:-0,1,2,3,4,5,6,7}"
EPGMM_EP16_DEVICES="${EPGMM_EP16_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
EPGMM_MASTER_PORT="${EPGMM_MASTER_PORT:-29616}"

# 超时针对每个用例；大规模默认给2小时。编译并行数与测试进程数无关。
EPGMM_TIMEOUT="${EPGMM_TIMEOUT:-1800}"
EPGMM_LARGE_TIMEOUT="${EPGMM_LARGE_TIMEOUT:-7200}"
EPGMM_BUILD_JOBS="${EPGMM_BUILD_JOBS:-4}"

# inherit：从安装前的主包元数据继承 timer。也可明确改为 ON/OFF。
EPGMM_TIMER="${EPGMM_TIMER:-inherit}"

# 每次自动建独立子目录，不覆盖旧结果。CACHE_ROOT 应在本机磁盘，不要放源码目录。
# 最大规模参考数据约需77 GiB临时空间，另留余量；还需检查设备显存和空闲状态。
EPGMM_RESULT_ROOT="${EPGMM_RESULT_ROOT:-/home/l00984991/project/result/epgmm_ep16}"
EPGMM_CACHE_ROOT="${EPGMM_CACHE_ROOT:-/home/l00984991/project/result/epgmm_ep16_cache}"
EPGMM_GOLDEN_BLOCK_ROWS="${EPGMM_GOLDEN_BLOCK_ROWS:-1024}"
EPGMM_BLAS_THREADS="${EPGMM_BLAS_THREADS:-1}"
# =========================== 配置区结束 ===============================

set -Ee -o pipefail

usage() {
    cat <<'USAGE'
用法：bash scripts/run_ep16_gmm.sh <build|accuracy|perf|plan|plan-perf>
  build          只编译安装和检查扩展，不运行设备测试
  accuracy       使用已安装包做两轮全输出精度验证（默认 ep8-baseline）
  perf           每例两轮精度通过后，以独立进程执行 msprof op_summary（默认 ep8-baseline）
  plan           只生成精度计划，不编译、不加载 CANN/设备扩展
  plan-perf      只生成精度和 msprof 性能计划，不执行设备用例
  无参数         显示帮助
  --build-only / --test-only / --plan 分别是 build / accuracy / plan 的兼容别名
  --help          显示说明

默认路径从脚本所在仓库推导。先准备依赖并完成 build，再运行 accuracy 或 perf。
默认先验证 EP8 的两个固定例；确认回归结果后，再显式选择 EP16 smoke/all。
ep8-baseline 的两个原生 smoke case 均为 G32、M131072、均衡路由、预分配输出：
  64d79f3ff1f3e5e6：alltoallv_gmm，K2048、N512
  84dbe9c348d7b2d6：gmm_alltoallv，K512、N2048
每例单独存入 ep8-baseline/<case-id>/result；默认使用设备0到7。
build/accuracy/perf 使用同目录检查器，检查 EP_GMM 全部可读 kernel 栈元数据，
超过32 KiB或缺失即停止；本次 ACL 配置存入结果目录，不修改site-packages。

perf 仅测 alltoallv_gmm 和 gmm_alltoallv，排除 composition 串联用例。
主指标为 kernel_us_per_call：逐调用跨 rank 取最大 kernel 耗时，再计算p50。
默认预热10次，采集5批、每批20次；两轮精度放在采集之外，不混入目标 kernel 计数。
MFU 使用每设备理论峰值 EPGMM_PEAK_TFLOPS（默认378 TFLOPS，当前950PR/28 Cube机器）。
性能表为 performance.csv，汇总为 summary.json 和 summary.md；失败项不填性能值。

路径、设备、套件、超时、结果目录在文件开头集中配置，也支持环境变量覆盖：
  bash scripts/run_ep16_gmm.sh build
  bash scripts/run_ep16_gmm.sh accuracy
  bash scripts/run_ep16_gmm.sh perf
  EPGMM_EP8_DEVICES=8,9,10,11,12,13,14,15 bash scripts/run_ep16_gmm.sh perf
  EPGMM_SUITES=smoke bash scripts/run_ep16_gmm.sh accuracy
  EPGMM_PERF_SUITES=smoke bash scripts/run_ep16_gmm.sh perf
  EPGMM_SUITES=all bash scripts/run_ep16_gmm.sh accuracy
  EPGMM_PERF_SUITES=all bash scripts/run_ep16_gmm.sh perf
  EPGMM_WARMUP=0 EPGMM_ITERATIONS=10 EPGMM_BATCHES=3 bash scripts/run_ep16_gmm.sh plan-perf

G/K/N、全局 T 和 TopK 网格在仓内
experiments/ep_gmm_fused/tests/framework/ep16_cases.py 中修改。
T 是16个rank合计的原始token数；每rank扩展行数 M=(T/16)*TopK。
smoke/topk/scale 的EP固定为16，改变设备列表长度不会改变测试的EP。
EP16 精度用例数：smoke=30、topk=450、scale=672、all=1152。
EP16 性能用例数：smoke=20、topk=300、scale=448、all=768；EP8默认仍为两例。
composition 是两个算子串联，不含激活、TopK加权合并或训练反向。
USAGE
}

die() {
    printf '错误：%s\n' "$*" >&2
    if [[ -n "${run_dir:-}" ]]; then
        printf 'failed: %s\n' "$*" > "$run_dir/status.txt"
        printf '日志：%s/run.log\n' "$run_dir" >&2
    fi
    exit 1
}
mode=
plan_only=false
measure=false
if (( $# > 1 )); then usage >&2; exit 2; fi
case "${1:-}" in
    '') usage; exit 0 ;;
    accuracy|--test-only) mode=accuracy ;;
    perf) mode=perf; measure=true ;;
    build|--build-only) mode=build ;;
    plan|--plan) mode=plan; plan_only=true ;;
    plan-perf) mode=plan-perf; plan_only=true; measure=true ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

suite_setting=EPGMM_SUITES
if [[ "$measure" == true ]]; then suite_setting=EPGMM_PERF_SUITES; fi
read -r -a suites <<< "${!suite_setting}"
(( ${#suites[@]} > 0 )) || die "$suite_setting 不能为空"
for suite in "${suites[@]}"; do
    case "$suite" in ep8-baseline|baseline|smoke|topk|scale|all) ;; *) die "未知套件：$suite" ;; esac
done
if [[ " ${suites[*]} " == *' all '* ]]; then
    for suite in "${suites[@]}"; do
        case "$suite" in smoke|topk|scale) die 'all 已包含 smoke/topk/scale，请勿重复组合' ;; esac
    done
fi
for setting in EPGMM_MASTER_PORT EPGMM_TIMEOUT EPGMM_LARGE_TIMEOUT EPGMM_BUILD_JOBS EPGMM_GOLDEN_BLOCK_ROWS EPGMM_BLAS_THREADS; do
    value="${!setting}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$setting 必须是正整数"
done
if [[ "$measure" == true ]]; then
    [[ "$EPGMM_WARMUP" =~ ^[0-9]+$ ]] || die 'EPGMM_WARMUP 必须是非负整数'
    for setting in EPGMM_ITERATIONS EPGMM_BATCHES; do
        value="${!setting}"
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$setting 必须是正整数"
    done
fi
(( EPGMM_MASTER_PORT <= 65535 && EPGMM_BUILD_JOBS <= 64 )) || die '端口须<=65535，编译并行数须<=64'
case "$EPGMM_TIMER" in inherit|ON|OFF) ;; *) die 'EPGMM_TIMER 只能是 inherit、ON 或 OFF' ;; esac
[[ -f "$EPGMM_REPO/setup.py" && -f "$EPGMM_REPO/experiments/ep_gmm_fused/tests/run.py" ]] || die "仓库目录不正确：$EPGMM_REPO"
[[ "$plan_only" == true || -f "$epgmm_stack_inspector" ]] || die "缺少仓内栈检查脚本：$epgmm_stack_inspector"
if [[ "$plan_only" != true ]]; then
    if [[ -n "${TORCH_ACL_INIT_CONFIG_PATH:-}" ]]; then
        [[ -f "$TORCH_ACL_INIT_CONFIG_PATH" ]] || die "原ACL配置文件不存在：$TORCH_ACL_INIT_CONFIG_PATH"
        export TORCH_ACL_INIT_CONFIG_PATH=$(cd "$(dirname "$TORCH_ACL_INIT_CONFIG_PATH")" && printf '%s/%s' "$PWD" "$(basename "$TORCH_ACL_INIT_CONFIG_PATH")")
    fi
fi
EPGMM_REPO=$(cd "$EPGMM_REPO" && pwd)
EPGMM_DEPS_REPO=$(cd "$EPGMM_DEPS_REPO" && pwd)

mkdir -p "$EPGMM_RESULT_ROOT" "$EPGMM_CACHE_ROOT"
run_dir=$(mktemp -d "$EPGMM_RESULT_ROOT/$(date +%Y%m%d_%H%M%S)_XXXXXX")
cache_dir="$EPGMM_CACHE_ROOT/$(basename "$run_dir")"
mkdir -p "$cache_dir"
# 将结果和缓存规范成绝对路径，后续切换目录不会改变输出位置。
run_dir=$(cd "$run_dir" && pwd)
cache_dir=$(cd "$cache_dir" && pwd)
exec > >(tee -a "$run_dir/run.log") 2>&1
on_error() {
    local rc="$1" line="$2"
    trap - ERR
    printf 'failed: exit=%s line=%s\n' "$rc" "$line" > "$run_dir/status.txt"
    printf '\n失败（退出码%s，第%s行），已停止后续步骤。日志：%s/run.log\n' "$rc" "$line" "$run_dir" >&2
    exit "$rc"
}
trap 'on_error "$?" "$LINENO"' ERR
printf 'running\n' > "$run_dir/status.txt"
printf '模式：%s\n结果：%s\n临时数据：%s\n套件：%s\n' "$mode" "$run_dir" "$cache_dir" "${suites[*]}"
if [[ "$measure" == true ]]; then
    printf 'msprof：每例两轮精度检查通过后，预热%s次，每批%s次，共%s批\n' "$EPGMM_WARMUP" "$EPGMM_ITERATIONS" "$EPGMM_BATCHES"
fi

if [[ -n "$EPGMM_CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$EPGMM_CONDA_ENV" ]]; then
    [[ -f "$EPGMM_CONDA_SH" ]] || die "找不到 $EPGMM_CONDA_SH；请修改配置或先激活 $EPGMM_CONDA_ENV"
    source "$EPGMM_CONDA_SH"
    conda activate "$EPGMM_CONDA_ENV"
fi
command -v "$EPGMM_PYTHON" >/dev/null || die "找不到 Python：$EPGMM_PYTHON"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
"$EPGMM_PYTHON" -I -c 'import sys, numpy; print("Python:", sys.executable); print("NumPy:", numpy.__version__)'

if [[ "$plan_only" != true ]]; then
    if [[ -n "$EPGMM_CANN_ENV" ]]; then
        [[ -f "$EPGMM_CANN_ENV" ]] || die "找不到 CANN 环境脚本：$EPGMM_CANN_ENV"
        source "$EPGMM_CANN_ENV"
    fi
    [[ -f "${ASCEND_HOME_PATH:-}/include/acl/acl.h" ]] || die '尚未加载 CANN；请设置文件开头的 EPGMM_CANN_ENV'
    export SHMEM_ROOT="$EPGMM_DEPS_REPO/third-party/shmem/install/shmem"
    export SHMEM_SOURCE_PATH="$EPGMM_DEPS_REPO/third-party/shmem"
    export CATCCOS_SOURCE_PATH="$EPGMM_DEPS_REPO/third-party/catccos"
    export CATLASS_ROOT="$EPGMM_DEPS_REPO/third-party/catlass"
    [[ -f "$SHMEM_ROOT/lib/libshmem.so" ]] || die "缺少原 SHMEM SDK：$SHMEM_ROOT"
    export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
fi

if [[ "$mode" == build ]]; then
    cd "$EPGMM_REPO"
    printf '\n===== 编译源码 =====\n'
    git branch --show-current
    git rev-parse HEAD
    if [[ "$EPGMM_TIMER" == inherit ]]; then
        epgmm_timer=$("$EPGMM_PYTHON" -I -c 'import importlib.metadata as m,json; p=m.distribution("ascend-deepep").locate_file("deep_ep/_build_info.json"); print("ON" if json.loads(p.read_text())["megamoe"]["timer"] else "OFF")')
    else
        epgmm_timer="$EPGMM_TIMER"
    fi
    export DEEPEP_MEGAMOE_TIMER="$epgmm_timer"
    export DEEPEP_BUILD_MEGAMOE=ON DEEPEP_BUILD_EP_GMM_FUSED=ON
    export DEEPEP_NPU_ARCH=Ascend950 CMAKE_BUILD_PARALLEL_LEVEL="$EPGMM_BUILD_JOBS"
    printf 'MegaMoE=ON，EP_GMM_FUSED=ON，timer=%s\n' "$epgmm_timer"
    "$EPGMM_PYTHON" -m pip install -v --force-reinstall --no-build-isolation --no-deps . 2>&1 | tee "$run_dir/build.log"
fi

# 扩展预检在结果目录运行，避免源码遮蔽安装包。仓内 worker 另行校验安装包来源。
cd "$run_dir"
if [[ "$plan_only" != true ]]; then
    # GM 版恢复32 KiB预算；实际需求以新编译的ELF为准，超限或元数据不完整即停止。
    # 配置只写入本次结果目录，保留原ACL设置，并在Torch初始化前导出。
    "$EPGMM_PYTHON" -I "$epgmm_stack_inspector" --max-stack-bytes 32768 --runtime-stack-bytes 32768 \
        --json "$run_dir/stack-metadata.json" --acl-config "$run_dir/acl-init.json"
    export TORCH_ACL_INIT_CONFIG_PATH="$run_dir/acl-init.json"
    "$EPGMM_PYTHON" -I - <<'PY'
import importlib.metadata as m
import json
import torch
import torch_npu
from deep_ep._native import require_native
import deep_ep_experimental.ep_gmm_fused._C as epgmm
import deep_ep_experimental.megamoe._C as megamoe
core = require_native("EP16 launcher")
info = json.loads(m.distribution("ascend-deepep").locate_file("deep_ep/_build_info.json").read_text())
assert info["megamoe"]["enabled"] and info["ep_gmm_fused"]["enabled"], "Installed wheel must contain both components"
print("Core:", core.__file__)
print("EP_GMM_FUSED:", epgmm.__file__)
print("MegaMoE:", megamoe.__file__)
print("MegaMoE timer:", info["megamoe"]["timer"])
PY
fi

if [[ "$mode" != build ]]; then
    cd "$EPGMM_REPO"
    native_mode=accuracy
    if [[ "$measure" == true ]]; then native_mode=perf; fi
    for suite in "${suites[@]}"; do
        devices="$EPGMM_EP16_DEVICES"; nproc=16; timeout="$EPGMM_TIMEOUT"
        native_suite="ep16-$suite"
        ep_filter=()
        case_ids=('')
        if [[ "$suite" == ep8-baseline ]]; then
            devices="$EPGMM_EP8_DEVICES"; nproc=8
            native_suite=smoke
            ep_filter=(--ep-size 8)
            case_ids=(64d79f3ff1f3e5e6 84dbe9c348d7b2d6)
        elif [[ "$suite" == baseline ]]; then
            devices="$EPGMM_BASELINE_DEVICES"; nproc=2
            native_suite=smoke
            ep_filter=(--ep-size 2)
        fi
        case "$suite" in topk|scale|all) timeout="$EPGMM_LARGE_TIMEOUT" ;; esac
        # 此处只校验列表，不探测或占用设备；设备是否空闲需由操作者安排。
        "$EPGMM_PYTHON" -I -c 'import sys; d=sys.argv[1].split(","); assert len(d)==int(sys.argv[2]) and all(x.isdecimal() for x in d) and len({int(x) for x in d})==len(d), f"Device list must have exactly {sys.argv[2]} distinct numeric IDs"' "$devices" "$nproc"
        for case_id in "${case_ids[@]}"; do
            output_dir="$run_dir/$suite"
            log_path="$run_dir/$suite.log"
            case_filter=()
            if [[ -n "$case_id" ]]; then
                output_dir="$output_dir/$case_id"
                log_path="$run_dir/${suite}_${case_id}.log"
                case_filter=(--case-id "$case_id")
            fi
            args=(-m experiments.ep_gmm_fused.tests.run "$native_mode" --suite "$native_suite" "${ep_filter[@]}" "${case_filter[@]}"
                --nnodes 1 --nproc-per-node "$nproc"
                --master-addr 127.0.0.1 --master-port "$EPGMM_MASTER_PORT" --timeout "$timeout"
                --reference-cache "$cache_dir" --output "$output_dir" --run-id result
                --golden-block-rows "$EPGMM_GOLDEN_BLOCK_ROWS" --blas-threads "$EPGMM_BLAS_THREADS")
            if [[ "$measure" == true ]]; then
                args+=(--warmup "$EPGMM_WARMUP" --iterations "$EPGMM_ITERATIONS" --batches "$EPGMM_BATCHES"
                    --peak-tflops "$EPGMM_PEAK_TFLOPS" --exclude-operator composition)
            fi
            if [[ "$plan_only" == true ]]; then args+=(--dry-run); fi
            printf '\n===== %s%s：EP%s，devices=%s，timeout=%ss =====\n' "$suite" "${case_id:+/$case_id}" "$nproc" "$devices" "$timeout"
            ASCEND_RT_VISIBLE_DEVICES="$devices" "$EPGMM_PYTHON" "${args[@]}" 2>&1 | tee "$log_path"
        done
    done
fi

if [[ "$plan_only" == true ]]; then
    printf 'plan_only\n' > "$run_dir/status.txt"
    printf '\n计划已生成，未进行设备测试，不能视为精度通过或性能结果。\n'
elif [[ "$mode" == build ]]; then
    printf 'build_completed\n' > "$run_dir/status.txt"
    printf '\n编译安装和扩展加载检查完成，未运行设备测试。\n'
else
    printf 'tests_passed\n' > "$run_dir/status.txt"
    if [[ "$measure" == true ]]; then
        printf '\n所选套件两轮精度与 msprof 采集均已完成。各套件摘要：%s/<suite>[/<case-id>]/result/summary.json\n' "$run_dir"
        printf '性能表：%s/<suite>[/<case-id>]/result/performance.csv\n' "$run_dir"
        printf '可读汇总：%s/<suite>[/<case-id>]/result/summary.md\n' "$run_dir"
    else
        printf '\n所选套件均已通过。各套件摘要：%s/<suite>[/<case-id>]/result/summary.json\n' "$run_dir"
    fi
fi
printf '完整日志：%s/run.log\n' "$run_dir"
