'''SI section "Fixation location and the final phase of fixation". Run: python -m fixloc.make.si.final_phase

Parts, run by name:
  amplifiers   the complete bipartite graphs with 100 vertices at r = 2, exact by lumping (seconds)
  n8           final phase against the value with every h(V minus v) set to 1, from the tables of the factorization
               section in ../factorization/raw/ (seconds)
  amp N [N ...]  amplification against fixation location statistics, from the same tables
'''
import sys

import numpy as np

from fixloc import allgraphs
from fixloc.bipartite import bipartite
from fixloc.make.si import dump, path

SECTION = 'final_phase'
TABLES = 'factorization'


def amplifiers(ms=(50, 20, 10, 1), N=100, r=2.0):
  return [bipartite(m, N - m, r) for m in ms]


def final_phase(N=8, r=2.0):
  '''Largest relative difference between the final phase and (r(N-1)+1)/r times the mean of 1/theta_v at the fixation
  location, the value of the final phase proposition with every h(V minus v) set to 1.'''
  g6s, cols, X = allgraphs.load(path(TABLES, f'raw/allgraphs_N{N}_r{r:g}.npz'))
  Tend = X[:, cols.index('Tend')]
  bound = (r * (N - 1) + 1) / r * X[:, cols.index('inv_th_last')]
  ratio = Tend / bound
  return dict(N=N, r=r, graphs=len(X), ratio_min=float(ratio.min()), ratio_max=float(ratio.max()))


def amp_location(N, rs=(1.25, 2.0, 10.0)):
  out = []
  for r in rs:
    _, _, X = allgraphs.load(path(TABLES, f'raw/allgraphs_N{N}_r{r:g}.npz'))
    out.append(allgraphs.amp_location(N, r, X))
  return out


if __name__ == '__main__':
  parts = sys.argv[1:] or ['amplifiers', 'n8']
  if 'amplifiers' in parts:
    dump(SECTION, 'amplifiers.json', amplifiers())
  if 'n8' in parts:
    dump(SECTION, 'final_phase_N8.json', final_phase())
  if parts[0] == 'amp':
    for N in parts[1:]:
      dump(SECTION, f'amp_location_N{N}.json', amp_location(int(N)))
