#!/bin/bash
#SBATCH -J charles-T
#SBATCH -p test
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 01:00:00
# Mutant sets at the expected fixation time T on the Charles River basin, networks 60 and 61 of fixloc.large.
# T is the expected fixation time conditional on fixation for a uniformly random initial vertex, from the main run.
# The run records the mutant sets at one checkpoint, round(T), with 4,000 fixation events per initial vertex as in the
# main run. Network 61 (r = 1) keeps the limit of 10^6 runs per initial vertex of run_network_capped.sh.
# The seed is 100 ID + 10 i + 4, with i the position of r in rlist.txt (run_network.sh uses + 1 to + 3).
# Run from the folder BASE that holds src/, after run_network.sh (network 60) and run_network_capped.sh (network 61).
# Writes ckT_r<r>.bin in each network folder and BASE/charles_watch_T.json.
set -euo pipefail
BASE=${BASE:-$PWD}
SRC=${SRC:-$BASE/src}
SIM=$SRC/fixloc/sim/sweepsim
PY=${PY:-python3}
export PYTHONPATH=$SRC
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
K=${K:-4000}
for ID in 60 61; do
  cd $BASE/large/net$ID
  if [ "$ID" = 61 ]; then CAP=1000000; else CAP=100000000; fi
  i=0
  for r in $(cat rlist.txt); do
    i=$((i + 1)); seed=$((ID * 100 + i * 10 + 4))
    T=$($PY -m fixloc.large expected main_r$r.bin)
    $SIM net.txt $r $K $seed ckT_r$r.bin $CAP $T
  done
done
ONLY=net60,net61 $PY -m fixloc.large watchT $BASE/large $BASE/charles_watch_T.json
