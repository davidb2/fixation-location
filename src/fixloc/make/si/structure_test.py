'''SI section "Testing a hypothesized population structure". Run: python -m fixloc.make.si.structure_test

Parts, run by name:
  fig4          the pair of Figure 4e to g, with the monitored vertices 3 and 6 for comparison (seconds)
  river         the river of Figure 2c with and without the directed edge, 100,000 sequences per structure (seconds)
  similar       pairs of graphs with N = 7 and similar fixation probability and fixation time (seconds)
  ratios        simulated counts over the bound, from the cluster tables raw/exact_test_r{r}.csv (seconds)
  exact R [M] [NPROC]   every graph with N = 7 against every rival with one edge added or removed (cluster, hours)
  location R    the same with the fixation location alone, with the Pearson test and the bound (cluster)
  pairs         the pair search over all graphs with N = 5 to 7 (cluster)
  alternatives  the 87 alternatives to the theorized structure of Figure 4 (cluster)
'''
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from fixloc import alternatives as alt
from fixloc import river
from fixloc import structure as st
from fixloc.census import graphs
from fixloc.exact import Chain, conditioned, location
from fixloc.make.si import dump, path

SECTION = 'structure_test'


def fig4_pair(extra=(3, 6), nmax=2000):
  '''The numbers of the pair of Figure 4, and the exact count for the monitored vertices 3 and 6.'''
  from fixloc.make import fig4

  ob = fig4.observable()
  cG, cH = st.chain(fig4.N, fig4.TRUE, fig4.R), st.chain(fig4.N, fig4.THEO, fig4.R)
  rG = st.last_occupied(conditioned(cG), cG.rho / cG.rho.sum(), cG.full, [extra])[0]
  rH = st.last_occupied(conditioned(cH), cH.rho / cH.rho.sum(), cH.full, [extra])[0]
  acc = st.binomial_accuracy(rG[0], rH[0], nmax)
  keys = ('monitored', 'P_true', 'P_theorized', 'first_n_095', 'stays_above_095_from', 'rho', 'first_n_095_fixation')
  out = {k: ob[k] for k in keys}
  out['P_location'] = [location(cG).tolist(), location(cH).tolist()]
  T = lambda c: float((c.rho * c.Tfix_moran).sum() / c.rho.sum())
  out['T'] = [T(cG), T(cH)]
  out['pair_3_6'] = dict(
    monitored=list(extra),
    P_true=rG.tolist(),
    P_theorized=rH.tolist(),
    first_n_095=next(i + 1 for i, a in enumerate(acc) if a >= 0.95),
  )
  return out


def river10(r=2.0, M=100000):
  '''Fixation events for a correct choice with probability 0.95 between the river with and without the directed edge.'''
  c1, c0 = Chain(river.graph(True), r), Chain(river.graph(False), r)
  L = np.clip(np.asarray(c1.L)[0], 0, None)
  P1, P0 = location(c1), location(c0)
  ne, nb = st.n_exact(P0, P1, M, np.random.default_rng([2026, 10, 200]))
  return dict(
    edges=river.EDGES,
    channel=river.CHANNEL,
    N=river.N,
    r=r,
    theta=np.asarray(c1.T, float).tolist(),
    L_mouth_r2=(L / L.sum()).tolist(),
    rho_mouth_r2=float(c1.rho[0]),
    P_with=P1.tolist(),
    P_without=P0.tolist(),
    n_exact=ne,
    n_bound=nb,
  )


def ratios(rs=(1.0, 2.0, 10.0)):
  '''Simulated count over the bound for the three observed quantities, from raw/exact_test_r{r}.csv.'''
  out = {}
  for r in rs:
    rows = list(csv.DictReader(open(path(SECTION, f'raw/exact_test_r{r:g}.csv'))))
    e = {}
    for q in ('location', 'origin_location', 'origin_time'):
      x = np.array([float(row[f'n_{q}']) / float(row[f'bound_{q}']) for row in rows])
      e[q] = dict(
        median=float(np.median(x)),
        q10=float(np.quantile(x, 0.1)),
        q90=float(np.quantile(x, 0.9)),
        share_0p5_to_0p7=float(np.mean((x >= 0.5) & (x <= 0.7))),
      )
    out[str(r)] = e
  return out


def exact(r, M=4000, nproc=8):
  gs = graphs(7)
  args = [(i, sorted(g.edges()), r, M) for i, g in enumerate(gs)]
  with Pool(nproc) as p:
    rows = [x for part in p.imap_unordered(st.exact_test_rows, args) for x in part]
  rows.sort()
  with open(path(SECTION, f'raw/exact_test_r{r:g}.csv'), 'w') as f:
    f.write(','.join(st.EXACT_COLS) + '\n')
    for x in rows:
      f.write('%d,%d,%d,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n' % x)
  dump(SECTION, f'exact_test_r{r:g}_summary.json', st.exact_test_summary(r, M, rows))


def location_only(r, nproc=8):
  gs = graphs(7)
  lam = st.gof_lambda()
  with Pool(nproc) as p:
    res = p.map(st.location_only_rows, [(i, sorted(g.edges()), r, lam) for i, g in enumerate(gs)], chunksize=4)
  with open(path(SECTION, f'raw/location_only_r{r:g}.csv'), 'w') as f:
    f.write('gid,a,b,kind,n_gof,n_lr,tv\n')
    for _, _, o in res:
      for x in o:
        f.write(','.join(str(v) for v in x) + '\n')
  dump(SECTION, f'location_only_r{r:g}.json', st.location_only_summary(r, res, gs, lam))


def alternatives(nproc=16):
  structs = [alt.H] + alt.alternatives()
  with Pool(nproc) as pool:
    out = pool.map(alt.records, structs)
  data = dict(H=alt.H, r=alt.R, pairs=alt.PAIRS, triples=alt.TRIPLES, structures=out)
  json.dump(data, open(path(SECTION, 'raw/fig4_alternatives.json'), 'w'))
  dump(SECTION, 'fig4_alternatives_summary.json', alt.summarize(out))


if __name__ == '__main__':
  parts = sys.argv[1:] or ['fig4', 'river', 'similar']
  if 'fig4' in parts:
    dump(SECTION, 'fig4_pair.json', fig4_pair())
  if 'river' in parts:
    dump(SECTION, 'river10.json', river10())
  if 'similar' in parts:
    dump(SECTION, 'similar_classical_N7.json', st.similar_pairs())
  if 'ratios' in parts:
    dump(SECTION, 'exact_test_ratios.json', ratios())
  if parts[0] == 'exact':
    exact(float(parts[1]), *(int(x) for x in parts[2:]))
  if parts[0] == 'location':
    location_only(float(parts[1]))
  if 'pairs' in parts:
    dump(SECTION, 'pair_search2.json', alt.pair_search(os.cpu_count()))
  if 'alternatives' in parts:
    alternatives()
