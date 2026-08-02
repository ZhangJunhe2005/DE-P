#!/usr/bin/env bash
# Clean a copied DE-P tree and create a source-focused archive for upload.
#
# Default input:
#   <login home>/下载/DE-P
#
# The original /home/zjh/YOPO/DE-P workspace is explicitly rejected.

set -Eeuo pipefail

MAX_ARCHIVE_BYTES=$((512 * 1024 * 1024))
MAX_RETAINED_REPORT_BYTES=$((5 * 1024 * 1024))

die() {
    echo "ERROR: $*" >&2
    exit 1
}

login_name="$(id -un)"
login_home="$(getent passwd "${login_name}" | cut -d: -f6)"
[[ -n "${login_home}" ]] || die "无法确定当前用户主目录"

default_copy="${login_home}/下载/DE-P"
copy_path="${1:-${default_copy}}"
copy_path="$(realpath -e -- "${copy_path}")"
original_path="$(realpath -e -- /home/zjh/YOPO/DE-P)"

[[ -d "${copy_path}" ]] || die "目标不是目录：${copy_path}"
[[ "${copy_path}" != "/" ]] || die "拒绝操作根目录"
[[ "${copy_path}" != "${login_home}" ]] || die "拒绝操作用户主目录"
[[ "${copy_path}" != "${original_path}" ]] || \
    die "拒绝清理原工程：${original_path}"
[[ "$(basename -- "${copy_path}")" == "DE-P" ]] || \
    die "目标目录名必须是 DE-P"
[[ "$(dirname -- "${copy_path}")" == "${login_home}/下载" ]] || \
    die "目标必须位于 ${login_home}/下载/DE-P"

# Project markers: require source files that identify this as the DE-P project.
for marker in train_dep.py policy/dep_network.py loss/loss_function.py; do
    [[ -f "${copy_path}/${marker}" ]] || \
        die "工程标志文件缺失：${copy_path}/${marker}"
done

archive_path="${login_home}/下载/DE-P-code-analysis.tar.gz"
manifest_path="${login_home}/下载/DE-P-code-analysis-manifest.txt"

echo "即将清理副本：${copy_path}"
echo "原工程受保护：${original_path}"
echo "输出压缩包：${archive_path}"
echo

# Large generated trees that are unnecessary for source-level analysis.
for relative in \
    data \
    cache \
    artifacts \
    runs \
    .git \
    .agents \
    .codex \
    reports/phase8c_map_determinism
do
    target="${copy_path}/${relative}"
    if [[ -e "${target}" ]]; then
        echo "删除生成目录：${relative}"
        rm -rf -- "${target}"
    fi
done

# Python/build caches and transient files.
find "${copy_path}" -depth -type d \
    \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \
       -o -name .ruff_cache \) \
    -exec rm -rf -- {} +

find "${copy_path}" -type f \
    \( -name '*.pyc' -o -name '*.pyo' -o -name '*.log' \
       -o -name '*.tmp' -o -name '*.swp' -o -name '*.swo' \
       -o -name 'core' -o -name 'core.*' \) \
    -delete

# Retain small reports/diagnostics, but remove bulky per-frame dumps.
for report_root in reports diagnostics; do
    if [[ -d "${copy_path}/${report_root}" ]]; then
        find "${copy_path}/${report_root}" -type f \
            -size +"${MAX_RETAINED_REPORT_BYTES}"c \
            -print -delete
    fi
done

# Catch accidental large binary/data files outside the known generated trees.
# Checkpoints in saved/ are deliberately retained.
find "${copy_path}" -type f -size +20M \
    ! -path "${copy_path}/saved/*" \
    -print -delete

{
    echo "DE-P ChatGPT code-analysis package"
    echo "Created: $(date --iso-8601=seconds)"
    echo "Source copy: ${copy_path}"
    echo
    echo "Top-level retained sizes:"
    du -h --max-depth=1 "${copy_path}" | sort -h
    echo
    echo "Retained files:"
    find "${copy_path}" -type f -printf '%P\n' | LC_ALL=C sort
} > "${manifest_path}"

rm -f -- "${archive_path}"
tar -C "$(dirname -- "${copy_path}")" \
    -czf "${archive_path}" \
    "$(basename -- "${copy_path}")"

archive_bytes="$(stat -c '%s' "${archive_path}")"
archive_human="$(du -h "${archive_path}" | cut -f1)"

if (( archive_bytes > MAX_ARCHIVE_BYTES )); then
    echo "压缩包大小：${archive_human}"
    die "压缩包仍超过 512 MiB；压缩包已保留，需进一步人工检查"
fi

echo
echo "UPLOAD_PACKAGE_RESULT"
echo "{"
echo "  \"status\": \"PASS\","
echo "  \"cleaned_copy\": \"${copy_path}\","
echo "  \"archive\": \"${archive_path}\","
echo "  \"archive_size_bytes\": ${archive_bytes},"
echo "  \"manifest\": \"${manifest_path}\","
echo "  \"original_workspace_modified\": false"
echo "}"
