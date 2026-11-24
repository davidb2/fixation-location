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
