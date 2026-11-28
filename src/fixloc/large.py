'''Simulated networks with N = 50 to 200 vertices: the networks, the analysis of the sweepsim output, and the summaries.

The cluster run has one folder large/net<ID> per network with net.txt, rlist.txt, and meta.json, written by networks().
src/fixloc/make/si/run_network.sh runs, for each fitness r, a pilot run for the three times, the main run, and a run
that records the mutant sets at the three times. The simulator and its reader are in fixloc.sim. Commands, from src/:
  python -m fixloc.large networks OUTDIR
  python -m fixloc.large quartiles PILOT.bin
  python -m fixloc.large analyze net.txt MAIN.bin CK.bin > summary_r<r>.json
  python -m fixloc.large gather BASE                (writes BASE/large_summary.json)
  python -m fixloc.large watch LARGE OUT.json       (writes the rank of the optimal vertex to monitor)
  python -m fixloc.large expected MAIN.bin          (prints round(T), T the expected fixation time given fixation)
  python -m fixloc.large watchT LARGE OUT.json      (writes the best vertex to sample at T, from ckT_r<r>.bin)
src/fixloc/make/si/run_charles_T.sh records the mutant sets at T on the Charles River basin, networks 60 and 61.
Only numpy is needed, except networkx for networks().
'''
import glob
import json
import os
import sys

import numpy as np

from fixloc.sim import edges, read_records, write_edges


def networks(out):
  '''Write the 60 networks: random 3-regular, Erdos-Renyi with mean degree 4, Barabasi-Albert with m = 2, Watts-Strogatz
  with k = 4 and p = 0.2, random trees, and random directed graphs (a directed cycle in random order plus one random
  outgoing edge per vertex), three of each at N = 50, 100, and 200, and a torus and a cycle at each size.'''
  import networkx as nx

  nets = []
  for size, side in ((50, 7), (100, 10), (200, 14)):
    for s in range(3):
      sd = 1000 * size + 10 * s
      nets.append(('random 3-regular', s, nx.random_regular_graph(3, size, seed=sd + 1)))
      k = 0
      while True:
        g = nx.gnp_random_graph(size, 4.0 / (size - 1), seed=sd + 2 + 100000 * k)
        if nx.is_connected(g):
          break
        k += 1
      nets.append(('Erdos-Renyi', s, g))
      nets.append(('Barabasi-Albert', s, nx.barabasi_albert_graph(size, 2, seed=sd + 3)))
      nets.append(('small-world', s, nx.connected_watts_strogatz_graph(size, 4, 0.2, seed=sd + 4)))
      nets.append(('random tree', s, nx.random_labeled_tree(size, seed=sd + 5)))
      rng = np.random.default_rng(sd + 6)
      perm = rng.permutation(size)
      D = nx.DiGraph()
      D.add_nodes_from(range(size))
      for i in range(size):
        D.add_edge(int(perm[i]), int(perm[(i + 1) % size]))
      for x in range(size):
        while True:
          y = int(rng.integers(size))
          if y != x and not D.has_edge(x, y):
            D.add_edge(x, y)
            break
      assert nx.is_strongly_connected(D)
      nets.append(('random directed', s, D))
    nets.append(('torus', 0, nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side, periodic=True))))
    nets.append(('cycle', 0, nx.cycle_graph(size)))
  for i, (fam, s, G) in enumerate(nets):
    d = os.path.join(out, f'net{i}')
    os.makedirs(d, exist_ok=True)
    N, E = edges(G)
    write_edges(N, E, os.path.join(d, 'net.txt'))
    open(os.path.join(d, 'rlist.txt'), 'w').write('1.5 2 5' + (' 1' if N <= 50 else '') + '\n')
    meta = dict(id=i, family=fam, instance=s, N=N, directed=G.is_directed(), links=len(E))
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'))
  return nets


def quartile_times(path):
  '''Time steps at which fixation has occurred with probability 0.25, 0.5, and 0.75, from a pilot run.

  The initial vertex is uniformly random and conditional on fixation, with weights from the simulated fixation
  probabilities.'''
  R = read_records(path)
  N = R['N']
  u, t = R['u'], R['t']
  rho = R['fixes'] / np.maximum(R['runs'], 1)
  pi = rho / rho.sum()
  n = np.bincount(u, minlength=N)
  w = pi[u] / np.maximum(n[u], 1)
  o = np.argsort(t)
  cw = np.cumsum(w[o])
  cw /= cw[-1]
  return [int(t[o][np.searchsorted(cw, q)]) for q in (0.25, 0.5, 0.75)]


def H2(p):
  p = np.clip(p, 1e-300, 1.0)
  q = np.clip(1.0 - p, 1e-300, 1.0)
  return -(p * np.log2(p) + q * np.log2(q))


def load_edges(path):
  with open(path) as f:
    N = int(f.readline())
    out = [[] for _ in range(N)]
    for line in f:
      a, b = map(int, line.split())
      if a != b:
        out[a].append(b)
  return N, out


def distances(N, out):
  '''D[u, v] is the number of steps from u to v along directed edges.'''
  D = np.full((N, N), np.inf)
  for s in range(N):
    D[s, s] = 0
    frontier = [s]
    d = 0
    while frontier:
      d += 1
      nxt = []
      for x in frontier:
        for y in out[x]:
          if D[s, y] == np.inf:
            D[s, y] = d
            nxt.append(y)
      frontier = nxt
  return D


def ranks_desc(x):
  o = np.argsort(-x, kind='stable')
  r = np.empty(len(x))
  r[o] = np.arange(1, len(x) + 1)
  return r


def spearman(a, b):
  if np.std(a) < 1e-12 or np.std(b) < 1e-12:
    return None
  return float(np.corrcoef(ranks_desc(a), ranks_desc(b))[0, 1])


def origin_guessing(R, D):
  '''Guesses of the initial vertex, scored on held-out runs ending in fixation.

  The runs from each initial vertex are split alternately into two halves. The first half estimates L(v | u), with 1/2
  added to every count, and the fixation time distribution in 20 bins with equal numbers of runs. The guesses are scored
  on the second half, with the initial vertex weighted by its fixation probability.'''
  N = R['N']
  u, v, t = R['u'], R['v'], R['t']
  rho = R['fixes'] / np.maximum(R['runs'], 1)
  pi = rho / rho.sum()
  first = np.searchsorted(u, np.arange(N))
  k = np.arange(len(u)) - first[u]
  tr, te = (k % 2 == 0), (k % 2 == 1)
  nte = np.bincount(u[te], minlength=N)
  w = pi[u[te]] / np.maximum(nte[u[te]], 1)
  ut, vt = u[te], v[te]
  a = 0.5
  C = np.zeros((N, N))
  np.add.at(C, (u[tr], v[tr]), 1)
  L = (C + a) / (C.sum(axis=1, keepdims=True) + a * N)
  post = pi[:, None] * L
  post /= post.sum(axis=0, keepdims=True)
  mapv = np.argmax(post, axis=0)
  acc_loc = float((w * (mapv[vt] == ut)).sum())
  HU = float(-(pi[pi > 0] * np.log2(pi[pi > 0])).sum())
  info_loc = HU - float((w * -np.log2(post[ut, vt])).sum())
  rk = np.vstack([ranks_desc(post[:, j]) for j in range(N)]).T
  mean_rank = float((w * rk[ut, vt]).sum())
  cand = (D == D.max(axis=0, keepdims=True)).astype(float)
  acc_far = float((w * cand[ut, vt] / cand.sum(axis=0)[vt]).sum())
  lt = np.log(t.astype(float))
  cuts = np.quantile(lt[tr], np.linspace(0, 1, 21)[1:-1])
  b = np.searchsorted(cuts, lt)
  Ct = np.zeros((N, 20))
  np.add.at(Ct, (u[tr], b[tr]), 1)
  Pt = (Ct + a) / (Ct.sum(axis=1, keepdims=True) + a * 20)
  pt = pi[:, None] * Pt
  pt /= pt.sum(axis=0, keepdims=True)
  acc_dur = float((w * (np.argmax(pt, axis=0)[b[te]] == ut)).sum())
  info_dur = HU - float((w * -np.log2(pt[ut, b[te]])).sum())
  nall = np.bincount(u, minlength=N)
  wall = pi[u] / np.maximum(nall[u], 1)
  last = np.bincount(v, weights=wall, minlength=N)
  out = dict(
    N=N,
    chance=1.0 / N,
    rho_mean=float(rho.mean()),
    prior_best=float(pi.max()),
    acc_location=acc_loc,
    acc_farthest=acc_far,
    acc_duration=acc_dur,
    info_location_bits=info_loc,
    info_duration_bits=info_dur,
    max_bits=float(np.log2(N)),
    mean_rank_location=mean_rank,
    takeovers=int(len(u)),
    test_takeovers=int(te.sum()),
    mean_duration=float((wall * t).sum()),
  )
  return out, pi, last


def watch_values(Rc, pi, last):
  '''Information in bits between fixation by t and the type of one or two vertices, averaged over the three times.'''
  N = Rc['N']
  u, t, S = Rc['u'], Rc['t'], Rc['state']
  n = np.bincount(u, minlength=N)
  w = pi[u] / np.maximum(n[u], 1)
  single = np.zeros(N)
  pair = np.zeros((N, N))
  fs, gains = [], []
  for k, tc in enumerate(Rc['checks']):
    f = float((w * (t <= tc)).sum())
    X = S[:, k, :].astype(float)
    m = w @ X
    single += H2(f) - m * H2(np.where(m > 0, f / np.maximum(m, 1e-300), 0))
    M2 = X.T @ (X * w[:, None])
    pair += H2(f) - M2 * H2(np.where(M2 > 0, f / np.maximum(M2, 1e-300), 0))
    fs.append(f)
    gains.append(f / np.maximum(m, 1e-300))
  single /= len(Rc['checks'])
  pair /= len(Rc['checks'])
  best = int(np.argmax(single))
  iu = np.triu_indices(N, 1)
  pv = pair[iu]
  bi = int(np.argmax(pv))
  top2 = tuple(sorted(np.argsort(-last)[:2].tolist()))
  return dict(
    checkpoints=[int(c) for c in Rc['checks']],
    p_complete_at_checkpoints=fs,
    best_node_rank_in_last=int((last > last[best]).sum() + 1),
    best_node_bits=float(single[best]),
    bits_at_most_likely_last=float(single[int(np.argmax(last))]),
    spearman_watch_vs_last=spearman(single, last),
    best_pair_bits=float(pv[bi]),
    top2_last_pair_bits=float(pair[top2]),
    p_complete_median=fs[1],
    p_complete_given_best_node_mutant_median=float(gains[1][best]),
    ck_takeovers=int(len(u)),
  )


def analyze(edges_path, main, ck=None):
  '''Summary of one network at one fitness, with the guesses of the initial vertex from the main run and monitoring
  from the run at three times.'''
  N, out = load_edges(edges_path)
  C, pi, last = origin_guessing(read_records(main), distances(N, out))
  if ck:
    C.update(watch_values(read_records(ck), pi, last))
  return C


def gather(base):
  '''Join large/net*/summary_r*.json with meta.json into base/large_summary.json.'''
  rows, missing = [], []
  folders = glob.glob(os.path.join(base, 'large', 'net*'))
  for d in sorted(folders, key=lambda p: int(p.rsplit('net', 1)[1])):
    meta = json.load(open(os.path.join(d, 'meta.json')))
    for r in open(os.path.join(d, 'rlist.txt')).read().split():
      f = os.path.join(d, f'summary_r{r}.json')
      try:
        s = json.load(open(f))
      except Exception:
        missing.append(f'{os.path.basename(d)} r={r}')
        continue
      rows.append(dict(meta, r=float(r), **s))
  json.dump(dict(rows=rows, missing=missing), open(os.path.join(base, 'large_summary.json'), 'w'))
  return rows, missing


def watch_rank(netdir, r, arrays=False):
  '''Rank of the optimal vertex to monitor among the vertices ordered by fixation location probability.

  The fixation location distribution comes from the main run. At each of the three times the optimal vertex has the
  largest f / m_w, from the run with the mutant sets.'''
  main = read_records(f'{netdir}/main_r{r}.bin')
  ck = read_records(f'{netdir}/ck_r{r}.bin')
  N = main['N']
  runs, fixes = main['runs'], main['fixes']
  pi = fixes / np.maximum(runs, 1)
  pi = pi / pi.sum()
  u, v = main['u'], main['v']
  n = np.bincount(u, minlength=N)
  last = np.zeros(N)
  np.add.at(last, v, pi[u] / np.maximum(n[u], 1))
  out = dict(N=N, takeovers=int(len(u)), ck_takeovers=int(len(ck['u'])))
  out.update(last_max=float(last.max()), last_argmax=int(np.argmax(last)))
  if arrays:
    out.update(last=last.tolist(), pi=pi.tolist(), fix_prob=(fixes / np.maximum(runs, 1)).tolist())
    out['mean_t'] = float(np.mean(main['t']))
  for k, tc in enumerate(ck['checks']):
    uc, t, S = ck['u'], ck['t'], ck['state'][:, k, :].astype(float)
    nc = np.bincount(uc, minlength=N)
    w = pi[uc] / np.maximum(nc[uc], 1)
    f = float((w * (t <= tc)).sum())
    m = w @ S
    F = np.where(m > 0, f / np.maximum(m, 1e-300), 0.0)
    best = int(np.argmax(F))
    rank = int((last > last[best] + 1e-12).sum() + 1)
    lk = int(np.argmax(last))
    out[f'ck{k}'] = dict(
      t=int(tc),
      f=f,
      best=best,
      rank=rank,
      F_best=float(F[best]),
      F_likeliest=float(F[lk]),
      m_best=float(m[best]),
      m_likeliest=float(m[lk]),
    )
    if arrays:
      out[f'ck{k}']['m'] = m.tolist()
  return out


def watch_ranks(large, outp, only=None, arrays=False):
  res = {}
  for netdir in sorted(glob.glob(os.path.join(large, 'net*'))):
    if only and os.path.basename(netdir) != only:
      continue
    nid = os.path.basename(netdir)[3:]
    if not os.path.exists(os.path.join(netdir, 'rlist.txt')):
      continue
    for r in open(os.path.join(netdir, 'rlist.txt')).read().split():
      if not (os.path.exists(f'{netdir}/main_r{r}.bin') and os.path.exists(f'{netdir}/ck_r{r}.bin')):
        continue
      try:
        res[f'{nid}:{r}'] = watch_rank(netdir, r, arrays)
      except Exception as e:
        res[f'{nid}:{r}'] = dict(error=str(e))
  json.dump(res, open(outp, 'w'), indent=1)
  return res


def location_law(main):
  '''Initial vertex given fixation, fixation location probabilities, and the expected fixation time T, for a uniformly
  random initial vertex and conditional on fixation, from the records of a main run.'''
  N = main['N']
  pi = main['fixes'] / np.maximum(main['runs'], 1)
  pi = pi / pi.sum()
  u, v, t = main['u'], main['v'], main['t']
  n = np.bincount(u, minlength=N)
  w = pi[u] / np.maximum(n[u], 1)
  last = np.bincount(v, weights=w, minlength=N)
  return pi, last, float((w * t).sum())


def expected_time(path):
  '''Expected fixation time T in time steps, conditional on fixation, for a uniformly random initial vertex, from a main
  run. T equals mean_duration of analyze().'''
  return location_law(read_records(path))[2]


def stratified_var(X, u, pi):
  '''Variance of the weighted mean of X over the runs ending in fixation, stratified by the initial vertex: the sum over
  initial vertices a of pi_a^2 Var_a(X) / n_a.'''
  var = np.zeros(X.shape[1:]) if X.ndim > 1 else 0.0
  for a in range(len(pi)):
    sel = u == a
    k = int(sel.sum())
    if pi[a] == 0 or k < 2:
      continue
    var = var + pi[a] ** 2 * X[sel].var(axis=0, ddof=1) / k
  return var


def sample_at(ck, k, pi, last):
  '''The best vertex to sample at checkpoint k of a run with mutant sets, with standard errors.

  m_w is the probability of a mutant at w at the checkpoint and f is the probability of fixation by the checkpoint, both
  conditional on fixation. The best vertex has the smallest m_w, which is the largest f / m_w. Its rank is its position
  among the vertices ordered by fixation location probability. z_gap is the difference of m_w between the second best
  and the best vertex in standard errors, and z_third the difference between the third best and the second best.'''
  N = len(pi)
  uc, t, S = ck['u'], ck['t'], ck['state'][:, k, :].astype(float)
  tc = int(ck['checks'][k])
  nc = np.bincount(uc, minlength=N)
  w = pi[uc] / np.maximum(nc[uc], 1)
  f = float((w * (t <= tc)).sum())
  m = w @ S
  se = np.sqrt(stratified_var(S, uc, pi))
  order = np.argsort(m, kind='stable')
  best, second, third = int(order[0]), int(order[1]), int(order[2])
  dvar = float(stratified_var(S[:, second] - S[:, best], uc, pi))
  tvar = float(stratified_var(S[:, third] - S[:, second], uc, pi))
  gap = float(m[second] - m[best])
  rank = lambda x: int((last > last[x] + 1e-12).sum() + 1)
  lk = int(np.argmax(last))
  lvar = float(stratified_var(S[:, lk] - S[:, best], uc, pi))
  top = [int(x) for x in np.argsort(-last, kind='stable')[:6]]
  return dict(
    t=tc, f=f, ck_fixations=int(len(uc)),
    best=best, rank=rank(best), m_best=float(m[best]), se_m_best=float(se[best]),
    second=second, rank_second=rank(second), m_second=float(m[second]),
    gap=gap, z_gap=gap / np.sqrt(dvar) if dvar > 0 else None,
    third=third, rank_third=rank(third), m_third=float(m[third]),
    z_third=float((m[third] - m[second]) / np.sqrt(tvar)) if tvar > 0 else None,
    likeliest=lk, m_likeliest=float(m[lk]), se_m_likeliest=float(se[lk]),
    likeliest_minus_best=float(m[lk] - m[best]),
    z_likeliest_minus_best=float((m[lk] - m[best]) / np.sqrt(lvar)) if lvar > 0 else None,
    F_best=f / float(m[best]), F_likeliest=f / float(m[lk]),
    top_locations=[dict(v=x, rank=rank(x), L=float(last[x]), m=float(m[x]), se_m=float(se[x])) for x in top],
    m=m.tolist(), se_m=se.tolist(),
  )


def watch_at_T(netdir, r):
  '''The best vertex to sample at the expected fixation time T of one network and fitness, from main_r<r>.bin and
  ckT_r<r>.bin, the run that records the mutant sets at round(T) (src/fixloc/make/si/run_charles_T.sh).'''
  main = read_records(f'{netdir}/main_r{r}.bin')
  pi, last, T = location_law(main)
  ck = read_records(f'{netdir}/ckT_r{r}.bin')
  assert len(ck['checks']) == 1 and abs(ck['checks'][0] - T) <= 1, (ck['checks'], T)
  return dict(net=int(os.path.basename(netdir)[3:]), r=float(r), N=int(main['N']), T=T,
              fixations_main=int(len(main['u'])), last=last.tolist(), pi=pi.tolist(), at_T=sample_at(ck, 0, pi, last))


def watch_at_T_all(large, outp, only=None):
  res = {}
  for netdir in sorted(glob.glob(os.path.join(large, 'net*'))):
    if only and os.path.basename(netdir) not in only.split(','):
      continue
    if not os.path.exists(os.path.join(netdir, 'rlist.txt')):
      continue
    nid = os.path.basename(netdir)[3:]
    for r in open(os.path.join(netdir, 'rlist.txt')).read().split():
      if os.path.exists(f'{netdir}/ckT_r{r}.bin'):
        res[f'{nid}:{r}'] = watch_at_T(netdir, r)
  json.dump(res, open(outp, 'w'), indent=1)
  return res


if __name__ == '__main__':
  cmd, args = sys.argv[1], sys.argv[2:]
  if cmd == 'networks':
    print(len(networks(args[0])), 'networks')
  elif cmd == 'quartiles':
    print(' '.join(str(x) for x in quartile_times(args[0])))
  elif cmd == 'analyze':
    print(json.dumps(analyze(*args)))
  elif cmd == 'gather':
    rows, missing = gather(args[0])
    print(len(rows), 'summaries;', len(missing), 'missing:', missing[:10])
  elif cmd == 'watch':
    res = watch_ranks(args[0], args[1], os.environ.get('ONLY'), bool(os.environ.get('ARRAYS')))
    print('wrote', args[1], len(res), 'cases')
  elif cmd == 'expected':
    print(round(expected_time(args[0])))
  elif cmd == 'watchT':
    res = watch_at_T_all(args[0], args[1], os.environ.get('ONLY'))
    print('wrote', args[1], len(res), 'cases')
