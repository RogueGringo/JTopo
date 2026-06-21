#!/usr/bin/env bash
# Launch a parallel, resumable powered-GUE ensemble on CPU.
#
#   bash run_powered_gue.sh <K> <N_TOTAL> <N_WORKERS> <OUT_BASE>
#
# Each worker is pinned to a single BLAS thread (so W workers ~ W cores) and
# writes its own shard, so the run is resume-safe and contention-free.
# Re-running the same command resumes: completed seeds are skipped per shard.
set -u

K="${1:-200}"
N_TOTAL="${2:-100}"
N_WORKERS="${3:-6}"
OUT="${4:-output/powered_gue_K${K}}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Interpreter: explicit $PYTHON override > repo-local venv (Windows/POSIX) > PATH.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/.venv-cpu/Scripts/python.exe" ]; then
  PY="$ROOT/.venv-cpu/Scripts/python.exe"
elif [ -x "$ROOT/.venv-cpu/bin/python" ]; then
  PY="$ROOT/.venv-cpu/bin/python"
else
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "${PY:-}" ] || ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "error: no Python interpreter found (set \$PYTHON, create .venv-cpu, or add python3 to PATH)" >&2
  exit 2
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "Launching $N_WORKERS workers: K=$K n_total=$N_TOTAL out=$OUT"
pids=()
for w in $(seq 0 $((N_WORKERS - 1))); do
  "$PY" "$ROOT/scripts/diagnostics/powered_gue_ensemble.py" \
      --K "$K" --n-total "$N_TOTAL" --worker-id "$w" --n-workers "$N_WORKERS" \
      --out "$OUT" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "All workers finished (fail=$fail)."
exit $fail
