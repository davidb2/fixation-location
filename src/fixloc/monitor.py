'''Monitoring for fixation on exact chains conditioned on fixation.

A mutant at the monitored vertex w at time t gives fixation by t with probability f_t / m_w(t), where f_t is the
probability of fixation by t and m_w(t) is the probability that w is mutant at t.
'''
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fixloc.exact import Chain, conditioned

EDGES12 = [
  (1, 2), (1, 4), (2, 1), (2, 10), (2, 11), (4, 5), (4, 7), (5, 6), (6, 8), (7, 9), (8, 10), (9, 3), (9, 6), (10, 5),
  (11, 12), (12, 6),
]


def watch_curves(c, p0, stop_at=0.76, after_median=0.0, tmax=200_000, min_t=0):
  '''Times, f_t and m_w(t) for every vertex w, from the distribution p0 of the chain conditioned on fixation.'''
  PT = conditioned(c)
  n = 2**c.N
  full = c.full
  member = ((np.arange(n)[:, None] >> np.arange(c.N)) & 1).astype(float)
  p = p0.copy()
  f = [p[full]]
  m = [p @ member]
  t = 0
  t_med = None
  while t < tmax:
    if t_med is None and f[-1] >= 0.5:
      t_med = t
    done = (t_med is not None and t >= (1 + after_median) * t_med) if after_median > 0 else f[-1] >= stop_at
    if done and t >= min_t:
      break
    p = PT @ p
    t += 1
    f.append(p[full])
    m.append(p @ member)
  return np.arange(t + 1), np.array(f), np.array(m)


def first_time(f, level):
  return int(np.argmax(f >= level))


def example12():
  '''Fixation location and monitoring curves at r = 1.25 on twelve vertices, initial mutant at vertex 1.'''
  G = nx.DiGraph()
  G.add_nodes_from(range(1, 13))
  G.add_edges_from(EDGES12)
  c = Chain(G, 1.25)
  N = c.N
  u = c.nodes.index(1)
  p0 = np.zeros(2**N)
  p0[1 << u] = 1.0
  t, f, m = watch_curves(c, p0, after_median=2.0)
  tm = {q: first_time(f, q) for q in (0.25, 0.5, 0.75)}
  L1 = np.clip(c.L[u], 0, None)
  out = dict(
    nodes=[int(x) for x in c.nodes],
    r=1.25,
    initial=1,
    edges=list(EDGES12),
    P_last=[float(x) for x in L1],
    theta=[float(x) for x in c.T],
    t_quartiles={str(q): int(v) for q, v in tm.items()},
    p_fix_given_mutant_at_median=[float(f[tm[0.5]] / m[tm[0.5], w]) for w in range(N)],
    p_mutant_at_median=[float(m[tm[0.5], w]) for w in range(N)],
    fixation_time_conditional=float(c.Tfix_moran[u]),
  )
  sel = np.unique(np.concatenate([np.arange(0, len(t), max(1, len(t) // 400)), [len(t) - 1]]))
  out['curves'] = dict(
    t=[int(x) for x in t[sel]],
    f=[float(x) for x in f[sel]],
    p_fix_given_mutant={str(c.nodes[w]): [float(f[i] / m[i, w]) if m[i, w] > 0 else None for i in sel] for w in range(N)},
  )
  return out


def islands_watch(n=10, m12=1e-3, m21=1e-1, rs=np.linspace(1.0, 1.5, 51), quantiles=(0.25, 0.5, 0.75, 0.9, 0.99)):
  '''Fixation by the sampling time given a mutant sampled from each island, from one mutant on island 1.'''
  idx = lambda a, b: a * (n + 1) + b
  S = (n + 1) ** 2
  full = idx(n, n)
  keys = [str(q) for q in list(quantiles) + ['exp']]
  out = dict(
    n=n,
    N=2 * n,
    m12=m12,
    m21=m21,
    r=[float(x) for x in rs],
    quantiles=[float(q) for q in quantiles],
    p1_last_island1=[],
    p_fix_given_mutant_island1={k: [] for k in keys},
    p_fix_given_mutant_island2={k: [] for k in keys},
    p_fix_given_island1_all_mutant={k: [] for k in keys},
    p_fix_given_island2_all_mutant={k: [] for k in keys},
    t_quantile={k: [] for k in keys},
    t_expected=[],
  )
  for r in rs:
    rows, cols, vals, vals_j = [], [], [], []
    B = np.zeros((S, 2))
    Bj = np.zeros((S, 2))
    lazy = np.zeros(S)
    for a in range(n + 1):
      for b in range(n + 1):
        if a + b in (0, 2 * n):
          continue
        k = idx(a, b)
        up1 = (a * r * (1 - m12) + b * r * m21) * (n - a) / n
        up2 = (a * r * m12 + b * r * (1 - m21)) * (n - b) / n
        dn1 = ((n - a) * (1 - m12) + (n - b) * m21) * a / n
        dn2 = ((n - a) * m12 + (n - b) * (1 - m21)) * b / n
        W = r * (a + b) + (2 * n - a - b)
        tot = up1 + up2 + dn1 + dn2
        lazy[k] = 1 - tot / W
        for rate, (a2, b2) in ((up1, (a + 1, b)), (up2, (a, b + 1)), (dn1, (a - 1, b)), (dn2, (a, b - 1))):
          if rate == 0:
            continue
          if a2 + b2 == 2 * n:
            col = 0 if a2 == a + 1 else 1
            B[k, col] += rate / W
            Bj[k, col] += rate / tot
          elif a2 + b2 > 0:
            rows.append(k)
            cols.append(idx(a2, b2))
            vals.append(rate / W)
            vals_j.append(rate / tot)
    Qj = sp.csr_matrix((vals_j, (rows, cols)), shape=(S, S))
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(S, S))
    X = spla.spsolve((sp.identity(S) - Qj).tocsc(), Bj)
    h = X.sum(axis=1)
    h[full] = 1.0
    p1 = X[idx(1, 0), 0] / h[idx(1, 0)]
    ok = h > 1e-14
    Qc = Q.tocoo()
    keep = ok[Qc.row] & ok[Qc.col]
    rr, cc = Qc.row[keep], Qc.col[keep]
    vv = Qc.data[keep] * h[cc] / h[rr]
    tofix = B.sum(axis=1) / np.where(ok, h, 1.0)
    src = np.flatnonzero(ok & (tofix > 0))
    rr = np.concatenate([rr, src])
    cc = np.concatenate([cc, np.full(len(src), full)])
    vv = np.concatenate([vv, tofix[src]])
    i_ok = np.flatnonzero(ok)
    stay = lazy.copy()
    stay[full] = 1.0
    P = sp.csr_matrix(
      (np.concatenate([vv, stay[i_ok]]), (np.concatenate([rr, i_ok]), np.concatenate([cc, i_ok]))), shape=(S, S)
    ).toarray()
    assert np.allclose(P[i_ok].sum(axis=1), 1.0), 'conditioned chain rows must sum to one'
    a_of = np.repeat(np.arange(n + 1), n + 1)
    b_of = np.tile(np.arange(n + 1), n + 1)
    p0 = np.zeros(S)
    p0[idx(1, 0)] = 1.0
    dist = lambda t: p0 @ np.linalg.matrix_power(P, t)
    trans = [k for k in i_ok if k != full]
    Q_t = P[np.ix_(trans, trans)]
    T_exp = float(np.linalg.solve(np.eye(len(trans)) - Q_t, np.ones(len(trans)))[trans.index(idx(1, 0))])
    out['t_expected'].append(T_exp)

    def time_at(level):
      hi = 1
      while dist(hi)[full] < level:
        hi *= 2
      lo = hi // 2
      while hi - lo > 1:
        mid = (lo + hi) // 2
        if dist(mid)[full] < level:
          lo = mid
        else:
          hi = mid
      return hi

    out['p1_last_island1'].append(float(p1))
    for q, key in zip(list(quantiles) + ['exp'], keys):
      t = int(round(T_exp)) if q == 'exp' else time_at(q)
      p = dist(t)
      f = p[full]
      m1 = p @ (a_of / n)
      m2 = p @ (b_of / n)
      c1 = p @ (a_of == n).astype(float)
      c2 = p @ (b_of == n).astype(float)
      out['p_fix_given_mutant_island1'][key].append(float(f / m1))
      out['p_fix_given_mutant_island2'][key].append(float(f / m2))
      out['p_fix_given_island1_all_mutant'][key].append(float(f / c1))
      out['p_fix_given_island2_all_mutant'][key].append(float(f / c2))
      out['t_quantile'][key].append(int(t))

  def cross(y1, y2):
    d = np.array(y1) - np.array(y2)
    s = np.flatnonzero(np.diff(np.sign(d)) != 0)
    if len(s) == 0:
      return None
    j = s[0]
    return float(rs[j] + (rs[j + 1] - rs[j]) * d[j] / (d[j] - d[j + 1]))

  for name, one, two in (
    ('crossing', 'p_fix_given_mutant_island1', 'p_fix_given_mutant_island2'),
    ('crossing_all_mutant', 'p_fix_given_island1_all_mutant', 'p_fix_given_island2_all_mutant'),
  ):
    out[name] = {k: cross(out[one][k], out[two][k]) for k in keys}
  out['r_star'] = cross(out['p1_last_island1'], [0.5] * len(rs))
  return out


QS = (0.25, 0.5, 0.75, 0.9, 'exp')


def watch_ranks(G, r, skip_at=0.5):
  '''Rank of the optimal vertex to monitor among the vertices ordered by fixation location probability.

  The initial vertex is uniformly random and the chain is conditioned on fixation. The optimal vertex has the largest
  f_t / m_w(t) at the expected fixation time ('exp') and at the first times with f_t at 0.25, 0.5, 0.75, and 0.9.
  Returns None if all vertices are equivalent at the time given by skip_at.
  '''
  c = Chain(G, r)
  N = c.N
  p0 = np.zeros(2**N)
  p0[1 << np.arange(N)] = c.rho / c.rho.sum()
  T_exp = float((c.rho * c.Tfix_moran).sum() / c.rho.sum())
  tT = int(round(T_exp))
  _, f, m = watch_curves(c, p0, stop_at=0.905, min_t=tT + 1)
  Pl = (c.rho / c.rho.sum()) @ np.clip(c.L, 0, None)
  order = np.argsort(-Pl)
  ts = tT if skip_at == 'exp' else first_time(f, skip_at)
  with np.errstate(divide='ignore', invalid='ignore'):
    ps = f[ts] / m[ts]
  if Pl.max() - Pl.min() < 1e-12 and ps.max() - ps.min() < 1e-12:
    return None
  row = dict(
    g6=nx.to_graph6_bytes(G, header=False).decode().strip() if not G.is_directed() else None,
    P_last_max=float(Pl.max()),
    t_median=first_time(f, 0.5),
    T_expected=T_exp,
    f_at_expected=float(f[tT]),
  )
  for q in QS:
    tq = tT if q == 'exp' else first_time(f, q)
    with np.errstate(divide='ignore', invalid='ignore'):
      pq = f[tq] / m[tq]
    best = pq.max()
    row[f'rank_best_{q}'] = int(np.flatnonzero(pq[order] >= best - 1e-12)[0]) + 1
    row[f'pf_best_{q}'] = float(best)
    row[f'pf_likeliest_{q}'] = float(pq[order[0]])
    row[f'pf_mean_{q}'] = float(np.mean(pq))
  row.update(rank_best=row['rank_best_0.5'], pf_best=row['pf_best_0.5'], pf_likeliest=row['pf_likeliest_0.5'])
  row['pf_random'] = row['pf_mean_0.5']
  return row


def rank_shares(rows):
  '''Shares of the graphs on which the optimal vertex is the likeliest or one of the two likeliest fixation
  locations.'''
  out = dict(evaluated=len(rows))
  for q in QS:
    rb = np.array([x[f'rank_best_{q}'] for x in rows])
    out[f'share_rank1_{q}'] = float(np.mean(rb == 1))
    out[f'share_rank_le2_{q}'] = float(np.mean(rb <= 2))
  return out


def binary_entropy(p):
  p = np.clip(p, 1e-300, 1)
  q = np.clip(1 - p, 1e-300, 1)
  return -(p * np.log2(p) + q * np.log2(q))


def information(f, m):
  '''Information in bits between fixation by t and the observation of mutants, where fixation implies the
  observation.'''
  return binary_entropy(f) - m * binary_entropy(np.where(m > 0, f / np.maximum(m, 1e-300), 0))


def watch_pairs(G, r, levels=(0.25, 0.5, 0.75)):
  '''Monitoring with one or two vertices, from a uniformly random initial vertex and conditional on fixation.

  At the first times with f_t at the levels, M2[a, b] is the probability of mutants at a and b at the same time. The
  information about fixation by t from the types at the monitored vertices is averaged over the levels.
  '''
  c = Chain(G, r)
  N, full = c.N, c.full
  prior = c.rho / c.rho.sum()
  x = np.zeros(2**N)
  x[1 << np.arange(N)] = prior
  M = ((np.arange(2**N)[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
  PT = conditioned(c)
  snaps, t = {}, 0
  while len(snaps) < len(levels):
    for q in levels:
      if q not in snaps and x[full] >= q:
        snaps[q] = (t, x.copy())
    x = PT @ x
    t += 1
  single, pair = np.zeros(N), np.zeros((N, N))
  iu = np.triu_indices(N, 1)
  least = {}
  for q, (t, x) in snaps.items():
    f = x[full]
    m = M.T @ x
    mm = M.T @ (x[:, None] * M)
    single += information(f, m) / len(levels)
    pair += information(f, mm) / len(levels)
    k = int(np.argmin(mm[iu]))
    least[str(q)] = dict(t=t, f=float(f), pair=[int(iu[0][k]), int(iu[1][k])], M2=float(mm[iu][k]))
  last = prior @ c.L
  bi = int(np.argmax(pair[iu]))
  top2 = tuple(sorted(np.argsort(-last)[:2].tolist()))
  return dict(
    nodes=[str(v) for v in c.nodes],
    r=float(r),
    P_last=last.tolist(),
    best_pair=[int(iu[0][bi]), int(iu[1][bi])],
    best_pair_bits=float(pair[iu][bi]),
    top2_last_pair=list(top2),
    top2_last_pair_bits=float(pair[top2]),
    least_mutant_pair=least,
  )
