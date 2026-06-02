#!/usr/bin/env bash
set -euo pipefail

RUNNER="./gemm_runner"
CSV_FILE="results/execution_time.csv"
SIZES=(256 512 1024 2048)
OMP_THREADS=6

if [[ ! -x "$RUNNER" ]]; then
    exit 1
fi

mkdir -p results/graphs test/graphs

echo "mode,size,time,time_sd,valid,gflops,t_h2d,t_kernel,t_d2h,gflops_kernel" > "$CSV_FILE"

for N in "${SIZES[@]}"; do
    $RUNNER --mode all --size "$N" --threads "$OMP_THREADS" --csv >> "$CSV_FILE"
done

cp -r results/* test/
