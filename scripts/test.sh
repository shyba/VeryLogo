#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT}/scripts/fmt.sh"
"${ROOT}/.venv/bin/python" -m unittest discover -s "${ROOT}/tests"

