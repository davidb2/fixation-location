'''The structure test. An observer chooses between a true and a theorized structure.

The initial vertex is uniformly random and every distribution is conditional on fixation.
The observer chooses the structure with the larger probability of all observations so far.
Ties count as half.
'''
import numpy as np
import networkx as nx
from scipy.stats import binom

from fixloc.exact import Chain, conditioned, location


def chain(N, edges, r):
  '''The exact chain of the undirected graph on the vertices 0..N-1 with the given edges.'''
  G = nx.Graph()
  G.add_nodes_from(range(N))
  G.add_edges_from(map(tuple, edges))
  return Chain(G, r)


def time_pmf(c, PT, tmax=4000):
  '''Fixation time distribution in time steps 1..tmax, with all longer times as the last entry.'''
  p = np.zeros(2**c.N)
  p[1 << np.arange(c.N)] = c.rho / c.rho.sum()
  f = [0.0]
  for _ in range(tmax):
    p = PT @ p
    f.append(p[c.full])
  return np.r_[np.diff(np.array(f)), max(0.0, 1 - f[-1])]


def summary(N, edges, r, tmax=4000):
  '''Fixation probability and time, fixation location and time distributions, and temperatures.'''
  c = chain(N, edges, r)
  w = c.rho / c.rho.sum()
  return dict(
    rho=float(c.rho.mean()),
    T=float((w * c.Tfix_moran).sum()),
    P=location(c),
    pmf=time_pmf(c, conditioned(c), tmax),
    theta=np.asarray(c.T, float),
  )


def last_occupied(PT, w, full, sets):
  '''For each set of monitored vertices, the distribution of the vertex last occupied by a wild type.'''
  N = len(w)
  n = PT.shape[0]
  X = np.arange(n)
  k = len(sets[0])
  S = len(sets)
  wild = np.stack([((X >> v) & 1) == 0 for v in range(N)], axis=1)
  only = np.full((n, S), -1)
  for s, M in enumerate(sets):
    wm = wild[:, list(M)]
    one = wm.sum(1) == 1
    only[one, s] = np.argmax(wm[one], axis=1)
  p = np.zeros((n, S, k))
  for u in range(N):
    x = 1 << u
    for s in range(S):
      p[x, s, max(only[x, s], 0)] += w[u]
  onehot = np.eye(k)
  for _ in range(100000):
    q = (PT @ p.reshape(n, S * k)).reshape(n, S, k)
    tot = q.sum(2)
    p = np.where((only >= 0)[:, :, None], onehot[np.maximum(only, 0)] * tot[:, :, None], q)
    if p[full].sum(1).min() > 1 - 1e-11:
      break
  return p[full] / p[full].sum(1, keepdims=True)


def simulated_accuracy(p, q, rng, sequences, nmax):
  '''Probability of choosing p over q after 1..nmax observations, from sequences simulated under p.'''
  llr = np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))
  draws = rng.choice(len(p), size=(sequences, nmax), p=p / p.sum())
  cum = np.cumsum(llr[draws], axis=1)
  return ((cum > 0) + 0.5 * (cum == 0)).mean(0)


def binomial_accuracy(p, q, nmax=200):
  '''The same for two possible observations with probabilities p and 1 - p, or q and 1 - q.'''
  out = []
  for n in range(1, nmax + 1):
    k = np.arange(n + 1)
    llr = k * np.log(p / q) + (n - k) * np.log((1 - p) / (1 - q))
    pk = binom.pmf(k, n, p)
    out.append(float(pk[llr > 1e-12].sum() + 0.5 * pk[np.abs(llr) <= 1e-12].sum()))
  return out


def outcome_accuracy(p, q, ns):
  '''The same from n trajectory outcomes, fixation or extinction, with fixation probabilities p and q.'''
  a, b = np.log(p / q), np.log((1 - p) / (1 - q))
  out = []
  for n in ns:
    ks = -n * b / (a - b)
    kf = np.floor(ks + 1e-12)
    tie = binom.pmf(round(ks), n, p) if abs(ks - round(ks)) < 1e-9 else 0.0
    out.append(float(binom.sf(kf, n, p) + 0.5 * tie))
  return out


def first_outcome_count(p, q, level=0.95, hi=10**7):
  '''Smallest number of trajectories with outcome_accuracy at least level, by bisection.'''
  lo = 1
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if outcome_accuracy(p, q, [mid])[0] >= level:
      hi = mid
    else:
      lo = mid
  return hi


def sample_events(PT, w, full, M, seed, m=20):
  '''Simulated fixation events, each shown at the last time step with a wild type at a vertex of M.'''
  N = len(w)
  P = PT.T.tocsr()
  rng = np.random.default_rng(seed)
  ev = []
  for _ in range(m):
    X = 1 << int(rng.choice(N, p=w))
    snap = X
    while X != full:
      a, b = P.indptr[X], P.indptr[X + 1]
      X = int(rng.choice(P.indices[a:b], p=P.data[a:b] / P.data[a:b].sum()))
      if any(((X >> v) & 1) == 0 for v in M):
        snap = X
    wild_m = [v for v in M if ((snap >> v) & 1) == 0]
    last = [v for v in range(N) if ((snap >> v) & 1) == 0]
    ev.append(dict(record=wild_m[0], mutants=[v for v in range(N) if (snap >> v) & 1], wild=last))
  return ev


def time_block(llr, cdf, ck, seed, m=125):
  '''Summed log likelihood ratios of m simulated sequences of fixation times, after each count in ck.'''
  rng = np.random.default_rng(seed)
  s = np.zeros(m)
  out = np.zeros((len(ck), m))
  done = 0
  k = 0
  while done < ck[-1]:
    step = min(10000, ck[-1] - done)
    idx = np.searchsorted(cdf, rng.random((m, step)), side='right')
    idx = np.minimum(idx, len(llr) - 1)
    cs = s[:, None] + np.cumsum(llr[idx], axis=1)
    while k < len(ck) and ck[k] <= done + step:
      out[k] = cs[:, ck[k] - done - 1]
      k += 1
    s = cs[:, -1]
    done += step
  return out


def similar_pairs(r=2.0, tol=0.01, N=7):
  '''Pairs of connected graphs with fixation probabilities and fixation times within tol of each other, compared by the
  total variation distance between their sorted fixation location distributions, a lower bound under any relabeling.'''
  from fixloc.census import graphs

  gs = graphs(N)
  R, T, S = [], [], []
  for g in gs:
    c = Chain(g, r)
    w = c.rho / c.rho.sum()
    R.append(float(c.rho.mean()))
    T.append(float((w * c.Tfix_moran).sum()))
    S.append(np.sort(w @ np.clip(c.L, 0, None))[::-1])
  R, T, S = np.array(R), np.array(T), np.array(S)
  dR = np.abs(R[:, None] - R[None, :]) / R[None, :]
  dT = np.abs(T[:, None] - T[None, :]) / T[None, :]
  TV = 0.5 * np.abs(S[:, None, :] - S[None, :, :]).sum(-1)
  close = (dR < tol) & (dT < tol)
  np.fill_diagonal(close, False)
  i, j = np.unravel_index(np.argmax(np.where(close, TV, -1)), TV.shape)
  example = dict(
    gids=[int(i), int(j)],
    rho=[R[i], R[j]],
    T=[T[i], T[j]],
    tv=float(TV[i, j]),
    signatures=[S[i].round(4).tolist(), S[j].round(4).tolist()],
    edges=[sorted(map(list, gs[i].edges())), sorted(map(list, gs[j].edges()))],
  )
  return dict(
    r=r,
    tolerance=tol,
    graphs=len(gs),
    close_pairs=int(close.sum() // 2),
    tv_max=float(TV[close].max()),
    share_tv_above_0p1=float((TV[close] > 0.1).mean()),
    share_tv_above_0p2=float((TV[close] > 0.2).mean()),
    example=example,
  )


def laws(G, r, tol=1e-11):
  '''Laws of the observation of one fixation event: the fixation location v, the pair (u, v), and the pair (t, u).'''
  c = Chain(G, r)
  F = c.time_distribution(tol=tol)
  F = F / F.sum()
  return F.sum(axis=(0, 1)), F.sum(axis=0).ravel(), F.sum(axis=2)


def pad(P, Q):
  '''Two laws of (t, u) on a common range of times, flattened.'''
  T = max(P.shape[0], Q.shape[0])
  if P.shape[0] < T:
    P = np.concatenate([P, np.zeros((T - P.shape[0],) + P.shape[1:])])
  if Q.shape[0] < T:
    Q = np.concatenate([Q, np.zeros((T - Q.shape[0],) + Q.shape[1:])])
  return P.ravel(), Q.ravel()


def bhattacharyya_n(P0, P1):
  '''Number of observations at which the bound on the probability of a wrong choice equals 0.05.'''
  B = -np.log(min(np.sqrt(np.clip(P0, 0, None) * np.clip(P1, 0, None)).sum(), 1.0))
  return float(np.log(10) / B) if B > 1e-15 else float('inf')


def n_exact(P0, P1, M, rng, level=0.05, block=256, cap_factor=2.0):
  '''Smallest n with a correct choice between P0 and P1 with probability at least 1 - level, from M sequences simulated
  under each law, ties counted as half. The search stops at cap_factor times the bound. Returns (n, bound).'''
  P0 = np.clip(np.asarray(P0, float), 0, None)
  P1 = np.clip(np.asarray(P1, float), 0, None)
  P0 = P0 / P0.sum()
  P1 = P1 / P1.sum()
  nb = bhattacharyya_n(P0, P1)
  if not np.isfinite(nb):
    return float('nan'), nb
  ok0, ok1 = P0 > 0, P1 > 0
  both = ok0 & ok1
  ell = np.zeros(P0.size)
  ell[both] = np.log(P1[both]) - np.log(P0[both])
  ell[ok1 & ~ok0] = 1e6
  ell[ok0 & ~ok1] = -1e6
  cdf0, cdf1 = np.cumsum(P0), np.cumsum(P1)
  cdf0[-1] = cdf1[-1] = 1.0
  S0, S1 = np.zeros(M), np.zeros(M)
  done = 0
  cap = int(np.ceil(cap_factor * nb)) + block
  while done < cap:
    k = min(block, cap - done)
    X0 = np.searchsorted(cdf0, rng.random((M, k)))
    X1 = np.searchsorted(cdf1, rng.random((M, k)))
    C0 = S0[:, None] + np.cumsum(ell[X0], axis=1)
    C1 = S1[:, None] + np.cumsum(ell[X1], axis=1)
    err = 0.5 * ((C0 > 0).mean(0) + 0.5 * (C0 == 0).mean(0) + (C1 < 0).mean(0) + 0.5 * (C1 == 0).mean(0))
    hit = np.flatnonzero(err <= level)
    if hit.size:
      return float(done + hit[0] + 1), nb
    S0, S1 = C0[:, -1], C1[:, -1]
    done += k
  return float('nan'), nb


def rivals(edges, N=7):
  '''Connected graphs obtained by adding or removing one edge, as (a, b, kind, graph).'''
  import itertools

  G = nx.Graph()
  G.add_nodes_from(range(N))
  G.add_edges_from(edges)
  for a, b in itertools.combinations(range(N), 2):
    H = G.copy()
    kind = 'remove' if H.has_edge(a, b) else 'add'
    H.remove_edge(a, b) if kind == 'remove' else H.add_edge(a, b)
    if nx.is_connected(H):
      yield a, b, kind, H


def exact_test_rows(args, seed=2026):
  '''Observed fixation events for a correct choice between a graph and each rival, for the fixation location, the
  initial vertex with the fixation location, and the initial vertex with the fixation time.'''
  gid, edges, r, M = args
  rng = np.random.default_rng([seed, gid, int(round(100 * r))])
  G = nx.Graph()
  G.add_nodes_from(range(7))
  G.add_edges_from(edges)
  L0, OL0, OT0 = laws(G, r)
  out = []
  for a, b, kind, H in rivals(edges):
    L1, OL1, OT1 = laws(H, r)
    A0, A1 = pad(OT0, OT1)
    res = [n_exact(P0, P1, M, rng) for P0, P1 in ((L0, L1), (OL0, OL1), (A0, A1))]
    out.append((gid, a, b, kind) + tuple(x for p in res for x in p))
  return out


EXACT_COLS = ['gid', 'a', 'b', 'kind']
EXACT_COLS += [f'{k}_{q}' for q in ('location', 'origin_location', 'origin_time') for k in ('n', 'bound')]


def exact_test_summary(r, M, rows):
  A = np.array([x[4:] for x in rows], float)
  nl, no, nt = A[:, 0], A[:, 2], A[:, 4]
  med = lambda x: float(np.nanmedian(x))
  return dict(
    r=r,
    M_per_truth=M,
    pairs=len(rows),
    censored=int(np.isnan(A[:, [0, 2, 4]]).sum()),
    median=dict(location=med(nl), origin_location=med(no), origin_time=med(nt)),
    share_location_beats_origin_time=float(np.mean(nl < nt)),
    share_location_beats_origin_location=float(np.mean(nl < no)),
    share_origin_location_beats_origin_time=float(np.mean(no < nt)),
    exact_over_bound_median=dict(location=med(nl / A[:, 1]), origin_time=med(nt / A[:, 5])),
  )


def location_law(G, r):
  '''Fixation location distribution with a uniformly random initial vertex, conditional on fixation.'''
  c = Chain(G, r)
  w = c.rho / c.rho.sum()
  return np.clip(w @ c.L, 0, None)


def gof_lambda(df=6, level=0.05, power=0.8):
  '''Noncentrality of the Pearson test with the given level and power.'''
  from scipy.optimize import brentq
  from scipy.stats import chi2, ncx2

  crit = chi2.ppf(1 - level, df)
  return brentq(lambda l: ncx2.sf(crit, df, l) - power, 0.1, 200)


def n_gof(P0, P1, lam):
  '''Observations for the Pearson test of P0 with power 0.8 when P1 is true, capped by the observations for one fixation
  event at a vertex that P0 excludes.'''
  forb = (P0 < 1e-12) & (P1 > 1e-12)
  ok = P0 >= 1e-12
  D = (((P1[ok] - P0[ok]) ** 2) / P0[ok]).sum()
  n = lam / D if D > 0 else np.inf
  if forb.any():
    n = min(n, np.log(5) / P1[forb].sum())
  return n


def n_lr(P0, P1):
  B = -np.log(min(np.sqrt(P0 * P1).sum(), 1.0))
  return np.log(10) / B if B > 0 else np.inf


def location_only_rows(args):
  '''The fixation location law of a graph and the test counts against each rival.'''
  gid, edges, r, lam = args
  G = nx.Graph()
  G.add_nodes_from(range(7))
  G.add_edges_from(edges)
  P0 = location_law(G, r)
  out = []
  for a, b, kind, H in rivals(edges):
    P1 = location_law(H, r)
    out.append((gid, a, b, kind, n_gof(P0, P1, lam), n_lr(P0, P1), float(np.abs(P0 - P1).sum() / 2)))
  return gid, P0, out


def location_only_summary(r, res, gs, lam):
  '''Medians of the test counts and the graphs that the sorted fixation location law does not separate.'''
  rows = [x for _, _, o in res for x in o]
  ng = np.array([x[4] for x in rows])
  nl = np.array([x[5] for x in rows])
  S = np.array([np.sort(P0) for _, P0, _ in res])
  D = np.abs(S[:, None, :] - S[None, :, :]).max(-1)
  np.fill_diagonal(D, np.inf)
  tied = np.where(D.min(1) < 1e-9)[0]
  vt = [i for i in range(len(gs)) if np.allclose(res[i][1], 1 / 7, atol=1e-9)]
  return dict(
    r=r,
    pairs=len(rows),
    lam=lam,
    n_gof_median=float(np.median(ng)),
    n_lr_median=float(np.median(nl)),
    n_gof_q=[float(np.quantile(ng, x)) for x in (0.1, 0.25, 0.75, 0.9)],
    frac_gof_le_100=float((ng <= 100).mean()),
    frac_inf=float(np.isinf(ng).mean()),
    graphs_tied=len(tied),
    graphs_uniform_law=len(vt),
    uniform_law_edges=[sorted(gs[i].edges()) for i in vt],
  )
