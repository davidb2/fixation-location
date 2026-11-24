#!/bin/bash
#SBATCH -J fixloc-large
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH --array=0-59
# One array task is one network of fixloc.large with all its fitness values. Per fitness r:
#   pilot (K_PILOT fixation trajectories per initial vertex) gives the three times of fixation probability 0.25, 0.5, 0.75,
#   main (K_MAIN) gives the fixation location and the guesses of the initial vertex,
#   ck (K_CK) records the mutant sets at the three times for monitoring.
# Setup: python -m fixloc.large networks $BASE/large, then make -C $SRC/fixloc/sim.
# After the array: python -m fixloc.large gather $BASE and python -m fixloc.large watch $BASE/large OUT.json.
set -euo pipefail
BASE=${BASE:-$PWD}
SRC=${SRC:-$BASE/src}
SIM=$SRC/fixloc/sim/sweepsim
PY=${PY:-python3}
export PYTHONPATH=$SRC
ID=${SLURM_ARRAY_TASK_ID:-$1}
K_PILOT=${K_PILOT:-100}; K_MAIN=${K_MAIN:-4000}; K_CK=${K_CK:-500}
RLIST=${RLIST:-$(cat $BASE/large/net$ID/rlist.txt)}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $BASE/large/net$ID
i=0
for r in $RLIST; do
  i=$((i + 1)); seed=$((ID * 100 + i * 10))
  $SIM net.txt $r $K_PILOT $((seed + 1)) pilot_r$r.bin
  CK=$($PY -m fixloc.large quartiles pilot_r$r.bin)
  $SIM net.txt $r $K_MAIN $((seed + 2)) main_r$r.bin
  $SIM net.txt $r $K_CK $((seed + 3)) ck_r$r.bin 100000000 $CK
  $PY -m fixloc.large analyze net.txt main_r$r.bin ck_r$r.bin > summary_r$r.json
  rm -f pilot_r$r.bin pilot_r$r.bin.meta
done
