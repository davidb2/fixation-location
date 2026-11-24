'''Exact summaries of every connected undirected graph with N vertices, from the graph list of nauty geng.

One row per graph, with the initial vertex uniformly random and conditional on fixation. The columns are COLS:
amp       fixation probability divided by the well-mixed value
Tm, Ta    fixation time in time steps and in active steps
Tend      time steps with exactly one wild type, Tend2 the same with one or two wild types
Tend_id   Tend from the final phase proposition, (r(N-1)+1)/r sum_v P(v) h(V minus v) / theta_v
inv_th_last, inv_th_mean   mean of 1/theta_v at the fixation location and over all vertices
h_last    mean of h(V minus v) at the fixation location
H, origin_last, MI         entropy of the fixation location, probability that the initial vertex is the fixation
                           location, and mutual information between the two, in bits
p_cold    probability of a fixation location of minimal temperature
Run on a cluster: python -m fixloc.allgraphs chunk TASK NTASKS G6FILE OUTDIR, then gather N G6FILE OUTDIR NTASKS.
The fitness values come from the environment variable FIXLOC_R, by default 1.25, 2, and 10.
'''
import glob
import gzip
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from fixloc.exact import Chain

R_VALUES = (
  tuple(float(x) for x in os.environ['FIXLOC_R'].split(',')) if os.environ.get('FIXLOC_R') else (1.25, 2.0, 10.0)
)
COLS = [
  'amp', 'Tm', 'Ta', 'Tend', 'Tend2', 'Tend_id', 'inv_th_last', 'inv_th_mean', 'h_last', 'H', 'origin_last', 'MI',
  'p_cold',
]


def geng(N):
  '''graph6 strings of the connected undirected graphs with N vertices.'''
  return subprocess.run(['geng', '-cq', str(N)], capture_output=True).stdout.split()


def read_g6(path):
  opener = gzip.open if path.endswith('.gz') else open
  with opener(path, 'rb') as f:
    return [line.strip() for line in f if line.strip()]


def summary(args):
  g6, r = args
  return summary_graph(nx.from_graph6_bytes(g6), r)


def summary_graph(G, r):
  '''One row of COLS for a graph G, undirected or directed with an outgoing edge at every vertex.'''
  c = Chain(G, r)
  N = c.N
  full = c.full
  rho = c.rho
  w = rho / rho.sum()
  L = np.clip(c.L, 0.0, None)
  P = w @ L
  th = np.asarray(c.T, float)
  Xv = np.array([full ^ (1 << v) for v in range(N)])
  hv = c.h[Xv]
  tm = c.G * c.h[:, None] * c.c[:, None]
  Tend_u = tm[Xv].sum(0) / rho
  near2 = c.nmut >= N - 2
  near2[full] = False
  Tend2_u = tm[near2].sum(0) / rho
  Tend = float(w @ Tend_u)
  Tend2 = float(w @ Tend2_u)
  Wl = r * (N - 1) + 1
  Tend_id = float(Wl / r * (P * hv / th).sum())
  J = w[:, None] * L
  pu = J.sum(1)
  pv = J.sum(0)
  nz = J > 1e-300
  MI = float((J[nz] * np.log2(J[nz] / np.outer(pu, pv)[nz])).sum())
  Pn = P[P > 1e-300]
  wm = 1.0 / N if r == 1 else (1 - 1 / r) / (1 - r**-N)
  return [
    rho.mean() / wm,
    float(w @ c.Tfix_moran),
    float(w @ c.Tfix_active),
    Tend,
    Tend2,
    Tend_id,
    float((P / th).sum()),
    float((1 / th).mean()),
    float((P * hv).sum()),
    float(-(Pn * np.log2(Pn)).sum()),
    float(np.trace(J)),
    MI,
    float(P[np.isclose(th, th.min())].sum()),
  ]


def compute(g6s, r, nproc=8):
  with Pool(nproc) as p:
    return np.array(p.map(summary, [(g, r) for g in g6s], chunksize=64), dtype=float)


def save(path, g6s, X):
  np.savez_compressed(path, g6=np.array(g6s), cols=np.array(COLS), X=X.astype(np.float64))


def load(path):
  '''graph6 strings, columns, and rows of a saved table.'''
  z = np.load(path, allow_pickle=True)
  return list(z['g6']), list(z['cols']), z['X']


def is_regular(g6s):
  return np.array([len(set(d for _, d in nx.from_graph6_bytes(g).degree())) == 1 for g in g6s])


def summarize(N, r, g6s, X):
  '''Summary of one table: rank correlations, shares of amplifiers and of graphs with a fixation location of low
  temperature, and the non-regular graphs whose fixation location has mean 1/theta_v at most the mean over all
  vertices.'''
  col = lambda k: X[:, COLS.index(k)]
  amp, Tm, Tend, ith, ithm = col('amp'), col('Tm'), col('Tend'), col('inv_th_last'), col('inv_th_mean')
  regular = is_regular(g6s)
  iso = np.isclose(ithm, 1.0)
  A = amp > 1 + 1e-9
  gap = ith - ithm
  exc = np.where((~regular) & (gap <= 1e-12))[0]
  g6 = lambda g: g.decode() if isinstance(g, bytes) else str(g)
  return dict(
    N=N,
    r=r,
    graphs=len(X),
    identity_max_rel_err=float(np.max(np.abs(col('Tend_id') - Tend) / Tend)),
    rho_amp_coldlast=float(spearmanr(amp, ith)[0]),
    rho_amp_T=float(spearmanr(amp, Tm)[0]),
    rho_coldlast_T=float(spearmanr(ith, Tm)[0]),
    rho_amp_Tend=float(spearmanr(amp, Tend)[0]),
    frac_amplifiers=float(A.mean()),
    frac_suppressors=float((amp < 1 - 1e-9).mean()),
    frac_last_colder_than_average=float((ith > ithm + 1e-9).mean()),
    frac_last_colder_than_average_among_amp=float((ith[A] > ithm[A] + 1e-9).mean()) if A.any() else None,
    share_end_min=float((Tend / Tm).min()),
    share_end_med=float(np.median(Tend / Tm)),
    share_end_max=float((Tend / Tm).max()),
    amp_max=float(amp.max()),
    T_ratio_maxamp_vs_isothermal=float(Tm[np.argmax(amp)] / np.median(Tm[iso])) if iso.any() else None,
    regular_graphs=int(regular.sum()),
    exceptions=[dict(g6=g6(g6s[i]), amp=float(amp[i]), gap=float(gap[i])) for i in exc],
  )


def amp_location(N, r, X):
  '''Rank correlations of the amplification with fixation location statistics, and medians by class of graph.'''
  col = lambda k: X[:, COLS.index(k)]
  amp = col('amp')
  reg = np.isclose(col('inv_th_mean'), 1.0)
  e = dict(
    N=N,
    r=r,
    graphs=int(len(X)),
    spearman={k: float(spearmanr(amp, col(k))[0]) for k in ('origin_last', 'H', 'p_cold', 'MI', 'inv_th_last', 'Tm')},
  )
  cls = {'suppressor': (amp < 1 - 1e-9) & ~reg, 'regular': reg, 'amplifier': (amp > 1 + 1e-9) & ~reg}
  cls['20 strongest amplifiers'] = np.isin(np.arange(len(amp)), np.argsort(amp)[-20:])
  e['other'] = int((~cls['suppressor'] & ~cls['regular'] & ~cls['amplifier']).sum())
  e['classes'] = {
    k: dict(count=int(m.sum()), **{q: float(np.median(col(q)[m])) for q in ('amp', 'origin_last', 'H', 'MI', 'p_cold')})
    for k, m in cls.items()
    if m.any()
  }
  return e


def directed_graphs(path):
  '''Edge lists of the strongly connected graphs in the output of nauty directg -T.'''
  graphs = []
  for line in open(path):
    x = [int(t) for t in line.split()]
    if not x:
      continue
    n, e = x[0], x[1]
    E = [(x[2 + 2 * i], x[3 + 2 * i]) for i in range(e)]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(E)
    if nx.is_strongly_connected(G):
      graphs.append(E)
  return graphs


def directed_row(args):
  edges, N, r = args
  G = nx.DiGraph()
  G.add_nodes_from(range(N))
  G.add_edges_from(edges)
  return summary_graph(G, r)


def directed_summary(N, mode, r, graphs, X):
  '''Share of the directed graphs with unequal temperatures whose fixation location has mean 1/theta_v at most the
  mean over all vertices.'''
  col = lambda k: X[:, COLS.index(k)]
  amp, ith, ithm, Tm = col('amp'), col('inv_th_last'), col('inv_th_mean'), col('Tm')
  iso = np.isclose(ithm, 1.0)
  gap = ith - ithm
  exc = np.where(~iso & (gap <= 1e-12))[0]
  ratio = ith / ithm
  return dict(
    N=N,
    mode=mode,
    r=r,
    graphs=len(X),
    isothermal=int(iso.sum()),
    non_isothermal=int((~iso).sum()),
    exceptions=int(len(exc)),
    frac_exceptions=float(len(exc) / max(1, (~iso).sum())),
    ratio_min_non_iso=float(ratio[~iso].min()),
    ratio_median_non_iso=float(np.median(ratio[~iso])),
    ratio_max=float(ratio.max()),
    identity_max_rel_err=float(np.max(np.abs(col('Tend_id') - col('Tend')) / col('Tend'))),
    frac_amplifiers=float((amp > 1 + 1e-9).mean()),
    frac_suppressors=float((amp < 1 - 1e-9).mean()),
    exceptions_amp_range=[float(amp[exc].min()), float(amp[exc].max())] if len(exc) else None,
    exceptions_ratio_min=float(ratio[exc].min()) if len(exc) else None,
    rho_amp_coldlast=float(spearmanr(amp, ith)[0]) if r != 1 else None,
    rho_amp_T=float(spearmanr(amp, Tm)[0]) if r != 1 else None,
    rho_coldlast_T=float(spearmanr(ith, Tm)[0]),
    exception_examples=[
      dict(edges=graphs[i], amp=float(amp[i]), ratio=float(ratio[i])) for i in exc[np.argsort(ratio[exc])][:10]
    ],
  )


def directed(path, N, out, rs=(1.0, 1.25, 2.0, 10.0), nproc=4):
  '''Exact rows and summaries for every strongly connected directed graph in a directg -T file.'''
  graphs = directed_graphs(path)
  summ, allX = [], {}
  for r in rs:
    with Pool(nproc) as p:
      X = np.array(p.map(directed_row, [(E, N, r) for E in graphs], chunksize=32), dtype=float)
    allX[f'r{r:g}'] = X
    summ.append(directed_summary(N, 'exhaustive', r, graphs, X))
  tag = f'directed_N{N}_exhaustive'
  np.savez_compressed(
    os.path.join(out, tag + '.npz'), cols=np.array(COLS), edges=np.array([json.dumps(E) for E in graphs]), **allX
  )
  json.dump(summ, open(os.path.join(out, tag + '_summary.json'), 'w'), indent=1)
  return summ


def chunk(task, ntask, src, out):
  '''One array task: the rows of a contiguous slice of a graph6 list, for every r in R_VALUES.'''
  g6 = read_g6(src)
  lo, hi = task * len(g6) // ntask, (task + 1) * len(g6) // ntask
  cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
  for r in R_VALUES:
    X = compute(g6[lo:hi], r, cpus)
    np.savez_compressed(os.path.join(out, f'chunk{task:03d}_r{r:g}.npz'), lo=lo, hi=hi, X=X)


def gather(N, src, out, ntask):
  '''Join the array output into allgraphs_N{N}_r{r}.npz and write allgraphs_N{N}_summary.json.'''
  g6 = read_g6(src)
  summ = []
  for r in R_VALUES:
    parts = [np.load(os.path.join(out, f'chunk{t:03d}_r{r:g}.npz')) for t in range(ntask)]
    assert [int(p['lo']) for p in parts] == [t * len(g6) // ntask for t in range(ntask)]
    X = np.concatenate([p['X'] for p in parts])
    assert len(X) == len(g6)
    save(os.path.join(out, f'allgraphs_N{N}_r{r:g}.npz'), g6, X)
    summ.append(summarize(N, r, g6, X))
  json.dump(summ, open(os.path.join(out, f'allgraphs_N{N}_summary.json'), 'w'), indent=1)
  return summ


def summarize_dir(folder):
  '''Summaries of every allgraphs_N*_r*.npz in a folder.'''
  out = []
  for f in sorted(glob.glob(os.path.join(folder, 'allgraphs_N*_r*.npz'))):
    N = int(f.split('_N')[1].split('_')[0])
    r = float(f.split('_r')[1][:-4])
    g6s, _, X = load(f)
    out.append(summarize(N, r, g6s, X))
  return out


if __name__ == '__main__':
  cmd, args = sys.argv[1], sys.argv[2:]
  if cmd == 'chunk':
    chunk(int(args[0]), int(args[1]), args[2], args[3])
  elif cmd == 'gather':
    gather(int(args[0]), args[1], args[2], int(args[3]))
  elif cmd == 'directed':
    directed(args[0], int(args[1]), args[2])
