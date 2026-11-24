'''SI section "Inferring the initial mutant location". Run: python -m fixloc.make.si.inference

The simulated networks with N = 50 to 200 run on a cluster with fixloc.large and run_network.sh, which write
large_summary.json. The part memory reads the tables of the factorization section for N = 8 (minutes).
'''
import sys

from fixloc import allgraphs, origin
from fixloc.make.si import dump, path

SECTION = 'inference'
TABLES = 'factorization'


def memory(N=8, rs=(1.25, 2.0, 10.0)):
  out = {}
  for r in rs:
    g6s, cols, X = allgraphs.load(path(TABLES, f'raw/allgraphs_N{N}_r{r:g}.npz'))
    out[str(r)] = origin.memory(g6s, X[:, cols.index('MI')], X[:, cols.index('amp')])
  return out


if __name__ == '__main__':
  if 'memory' in (sys.argv[1:] or ['memory']):
    dump(SECTION, 'origin_memory_N8.json', memory())
