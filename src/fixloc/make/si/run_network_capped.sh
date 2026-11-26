#!/bin/bash
#SBATCH -J fixloc-capped
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH --array=61
# run_network.sh with a cap of MAX_RUNS runs per initial vertex in all three phases (pilot, main, ck).
# Used for the neutral run on the Charles River basin, network 61 with r = 1 (see src/fixloc/make/si/charles_basin.py).
# Without a cap, sweepsim stops after K x 100,000 runs per initial vertex in the pilot and main phases. Under neutral
# evolution a mutant at a vertex near the root of the basin reaches fixation fewer than 74 times in 10^7 runs, so almost
# all the time goes to initial vertices with negligible weight for a uniformly random initial vertex.
# With the default MAX_RUNS of 10^6 the run took 23 minutes on 4 cores.
set -euo pipefail
BASE=${BASE:-$PWD}
SRC=${SRC:-$BASE/src}
SIM=$SRC/fixloc/sim/sweepsim
PY=${PY:-python3}
export PYTHONPATH=$SRC
ID=${SLURM_ARRAY_TASK_ID:-$1}
K_PILOT=${K_PILOT:-100}; K_MAIN=${K_MAIN:-4000}; K_CK=${K_CK:-500}; MAX_RUNS=${MAX_RUNS:-1000000}
RLIST=${RLIST:-$(cat $BASE/large/net$ID/rlist.txt)}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $BASE/large/net$ID
i=0
for r in $RLIST; do
  i=$((i + 1)); seed=$((ID * 100 + i * 10))
  $SIM net.txt $r $K_PILOT $((seed + 1)) pilot_r$r.bin $MAX_RUNS
  CK=$($PY -m fixloc.large quartiles pilot_r$r.bin)
  $SIM net.txt $r $K_MAIN $((seed + 2)) main_r$r.bin $MAX_RUNS
  $SIM net.txt $r $K_CK $((seed + 3)) ck_r$r.bin $MAX_RUNS $CK
  $PY -m fixloc.large analyze net.txt main_r$r.bin ck_r$r.bin > summary_r$r.json
  rm -f pilot_r$r.bin pilot_r$r.bin.meta
done
