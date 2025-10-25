#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./run_app.sh

Creates (or reuses) .venv_unix_pyXY inside the repository, installs dependencies,
and launches Streamlit with the detected Python 3.9-3.13 interpreter.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ensure_python() {
  local candidates=()
  if [[ -n "${PYTHON:-}" ]]; then
    candidates+=("$PYTHON")
  fi
  candidates+=("python3" "python")

  for cmd in "${candidates[@]}"; do
    if command -v "$cmd" >/dev/null 2>&1; then
      local info
      if info=$("$cmd" -c 'import sys, pathlib; print(f"{sys.version_info.major}.{sys.version_info.minor};{pathlib.Path(sys.executable).resolve()}")' 2>/dev/null); then
        PY_VERSION="${info%%;*}"
        PYTHON_EXE="${info#*;}"
        return 0
      fi
    fi
  done
  return 1
}

if ! ensure_python; then
  cat >&2 <<'EOF'
ERROR: No suitable Python interpreter found.
Install Python 3.9 - 3.13 (https://www.python.org/downloads/) and ensure it is on your PATH.
EOF
  exit 1
fi

major="${PY_VERSION%%.*}"
minor="${PY_VERSION#*.}"

if [[ "$major" != "3" ]]; then
  echo "ERROR: Detected Python ${PY_VERSION}, but this project requires Python 3.9 - 3.13." >&2
  exit 1
fi
if (( minor < 9 )); then
  echo "ERROR: Python ${PY_VERSION} is too old. Install Python 3.9 - 3.13." >&2
  exit 1
fi
if (( minor > 12 )); then
  cat >&2 <<'EOF'
ERROR: Python 3.13 detected. The dependencies (matplotlib, lxml) do not yet ship wheels for 3.13.
Install Python 3.9 - 3.12 and rerun ./run_app.sh.
EOF
  exit 1
fi

VENV_DIR="${PROJECT_DIR}/.venv_unix_py${major}${minor}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtual environment at ${VENV_DIR} ..."
  "${PYTHON_EXE}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

exec python -m streamlit run app.py
