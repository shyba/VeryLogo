#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLACK="${ROOT}/.venv/bin/black"

if [[ ! -x "${BLACK}" ]]; then
  echo "black not found at ${BLACK}"
  exit 1
fi

while IFS= read -r -d '' file; do
  "${BLACK}" -q "${file}"
done < <(
  find "${ROOT}/stc" "${ROOT}/tests" "${ROOT}/scripts" -type f -name '*.py' -print0 | sort -z
)
