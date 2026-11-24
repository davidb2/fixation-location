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
