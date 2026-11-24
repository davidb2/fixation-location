'''SI section "Which vertex to monitor to learn whether fixation has occurred". Run: python -m fixloc.make.si.monitoring

Parts, run by name, for example python -m fixloc.make.si.monitoring fig1 pairs:
  fig1    temperatures and fixation probabilities of the source and sink structure of Figure 1 (about a minute)
  pairs   the pair of monitored vertices on the cycle, the 2 x 5 grid, and a random 3-regular graph, N = 10 (seconds)
  all7    every connected undirected graph with N = 7 at r = 1.25, 2, and 10 (minutes)
  all9    every connected undirected graph with N = 9, one r per call, on a cluster: all9 R G6FILE [NPROC]
The simulated networks with N = 50 to 200 come from fixloc.sim.large (cluster), which writes large_watch_rank.json.
'''
import os
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np

from fixloc import monitor, origin
from fixloc.allgraphs import read_g6
from fixloc.census import graphs
from fixloc.exact import Chain
from fixloc.make import fig1
from fixloc.make.si import dump, path

SECTION = 'monitoring'
SOURCES = (0, 1, 2)


def fig1_sources(r=1.0):
  c = Chain(fig1.graph(), r)
  return dict(r=r, sources=list(SOURCES), theta=np.asarray(c.T, float).tolist(), rho=c.rho.tolist())


def pairs10(rs=(1.0, 2.0, 5.0)):
  '''Monitoring pairs on the cycle, the grid, and the random 3-regular graph of the N = 10 families of fixloc.origin.'''
  out = []
  for fam, s, G in origin.families(10, seeds=1):
    if fam not in ('cycle', 'grid', 'random 3-regular'):
      continue
    nodes = sorted(G.nodes())
    d = dict(nx.all_pairs_shortest_path_length(G))
    D = [[d[a][b] for b in nodes] for a in nodes]
    for r in rs:
      out.append(dict(family=fam, distances=D, **monitor.watch_pairs(G, r)))
  return out


def all7(rs=(1.25, 2.0, 10.0)):
  gs = graphs(7)
  rows, summ = {}, {}
  for r in rs:
    rows[str(r)] = [x for x in (monitor.watch_ranks(G, r) for G in gs) if x is not None]
    summ[str(r)] = dict(r=r, graphs=len(gs), skipped=len(gs) - len(rows[str(r)]), **monitor.rank_shares(rows[str(r)]))
  return rows, summ


def all9_row(args):
  g6, r = args
  return monitor.watch_ranks(nx.from_graph6_bytes(g6), r, skip_at='exp')


def all9(r, g6file, nproc=8):
  '''Rows and shares for every graph of a graph6 file, with monitor_N9_r{r}.csv in raw/.'''
  g6s = read_g6(g6file)
  with Pool(nproc) as p:
    res = p.map(all9_row, [(g, r) for g in g6s], chunksize=64)
  rows = [dict(x, g6=g.decode()) for x, g in zip(res, g6s) if x is not None]
  cols = ['g6', 'T_expected', 'f_at_expected']
  cols += [f'{k}_{q}' for q in monitor.QS for k in ('rank_best', 'pf_best', 'pf_likeliest', 'pf_mean')]
  with open(path(SECTION, f'raw/monitor_N9_r{r:g}.csv'), 'w') as f:
    f.write(','.join(cols) + '\n')
    for x in rows:
      f.write(','.join(str(x[c]) if not isinstance(x[c], float) else f'{x[c]:.6g}' for c in cols) + '\n')
  summ = dict(r=r, graphs=len(g6s), evaluated=len(rows), all_equivalent_skipped=len(g6s) - len(rows))
  summ.update({k: v for k, v in monitor.rank_shares(rows).items() if k != 'evaluated'})
  dump(SECTION, f'monitor_N9_r{r:g}_summary.json', summ)
  return summ


if __name__ == '__main__':
  parts = sys.argv[1:] or ['fig1', 'pairs']
  if 'fig1' in parts:
    dump(SECTION, 'fig1_sources.json', fig1_sources())
  if 'pairs' in parts:
    dump(SECTION, 'pairs_N10.json', pairs10())
  if 'all7' in parts:
    rows, summ = all7()
    dump(SECTION, 'raw/watch_all7.json', rows)
    dump(SECTION, 'watch_all7_summary.json', summ)
  if parts[0] == 'all9':
    all9(float(parts[1]), parts[2], int(parts[3]) if len(parts) > 3 else os.cpu_count())
