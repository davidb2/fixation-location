'''Exact solver for Birth-death updating on a directed graph.

States are mutant sets stored as bitmasks 0 .. 2^N - 1. State 0 is extinction and 2^N - 1 is fixation.
Moran steps are the time steps of the paper. Active steps are the steps that change the mutant set.
'''

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx


def as_digraph(G):
  '''Relabel nodes 0..N-1 (sorted order) and return out-neighbor lists.'''
  nodes = sorted(G.nodes())
  idx = {u: i for i, u in enumerate(nodes)}
  N = len(nodes)
  out = [[] for _ in range(N)]
  if G.is_directed():
    for a, b in G.edges():
      if a != b:
        out[idx[a]].append(idx[b])
  else:
    for a, b in G.edges():
      if a != b:
        out[idx[a]].append(idx[b])
        out[idx[b]].append(idx[a])
  out = [sorted(set(o)) for o in out]
  return N, out, nodes


def temperature(N, out):
  '''T_v = sum over in-neighbors y of 1/outdeg(y) (Lieberman, Hauert, Nowak 2005).'''
  T = np.zeros(N)
  for y in range(N):
    for v in out[y]:
      T[v] += 1.0 / len(out[y])
  return T


class Chain:
  def __init__(self, G, r):
    self.r = float(r)
    self.N, self.out, self.nodes = as_digraph(G)
    N, out, r = self.N, self.out, self.r
    if N > 16:
      raise ValueError('exact solver is meant for N <= 16')
    S = np.arange(2**N, dtype=np.int64)
    self.S = S
    self.full = 2**N - 1
    self.nmut = np.array([bin(s).count('1') for s in range(2**N)])
    self.W = r * self.nmut + (N - self.nmut)
    rows, cols, vals = [], [], []
    a = np.zeros(2**N)
    for x in range(N):
      dx = len(out[x])
      if dx == 0:
        continue
      bx = 1 << x
      for y in out[x]:
        by = 1 << y
        xm = (S & bx) != 0
        ym = (S & by) != 0
        m = xm & ~ym
        rows.append(S[m])
        cols.append(S[m] | by)
        vals.append(np.full(m.sum(), r / dx))
        m = ~xm & ym
        rows.append(S[m])
        cols.append(S[m] & ~by)
        vals.append(np.full(m.sum(), 1.0 / dx))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    np.add.at(a, rows, vals)
    a[0] = 0.0
    a[self.full] = 0.0
    keep = (rows != 0) & (rows != self.full)
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    self.a = a
    self.R = sp.csr_matrix((vals, (rows, cols)), shape=(2**N, 2**N))
    self.R.sum_duplicates()
    active = a > 0
    self.active = active
    inv_a = np.where(active, 1.0 / np.where(active, a, 1.0), 0.0)
    self.Q = sp.diags(inv_a) @ self.R
    self.M = (sp.identity(2**N, format='csc') - self.Q).tocsc()
    self.lu = spla.splu(self.M)
    self.T = temperature(N, out)
    self._solve()

  def _solve(self):
    N, full = self.N, self.full
    B = np.zeros((2**N, N))
    for v in range(N):
      s = full ^ (1 << v)
      if self.a[s] > 0:
        B[s, v] = self.R[s, full] / self.a[s]
    self.X = self.lu.solve(B)
    self.h = self.X.sum(axis=1)
    starts = np.array([1 << u for u in range(N)])
    self.rho = self.h[starts]
    with np.errstate(invalid='ignore', divide='ignore'):
      self.L = self.X[starts, :] / self.rho[:, None]
    E = np.zeros((2**N, N))
    E[starts, np.arange(N)] = 1.0
    self.G = self.lu.solve(E, trans='T')
    with np.errstate(invalid='ignore', divide='ignore'):
      self.c = np.where(self.active, self.W / np.where(self.active, self.a, 1.0), 0.0)
    wt = self.G * self.h[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
      self.Tfix_active = wt.sum(axis=0) / self.rho
      self.Tfix_moran = (wt * self.c[:, None]).sum(axis=0) / self.rho

  def time_distribution(self, prior=None, tol=1e-10, tmax=10**7, chunk=None):
    '''F[t, u, v] = P(fixation at Moran step t + 1 with last wild type v | start u), until mass < tol.'''
    N, full = self.N, self.full
    Wv = self.W.astype(float)
    P = (sp.diags(1.0 / Wv) @ self.R).tocsr()
    stay = 1.0 - self.a / Wv
    PT = P.T.tocsr()
    x = np.zeros((2**N, N))
    x[[1 << u for u in range(N)], np.arange(N)] = 1.0
    last = [full ^ (1 << v) for v in range(N)]
    rate_last = np.array([self.R[s, full] / Wv[s] for s in last])
    out = []
    t = 0
    while t < tmax:
      out.append(x[last, :].T * rate_last[None, :])
      x = PT @ x + stay[:, None] * x
      x[0, :] = 0.0
      x[full, :] = 0.0
      stuck = ~self.active
      x[stuck, :] = 0.0
      t += 1
      if t % 256 == 0 and x.sum(axis=0).max() < tol:
        break
    return np.array(out)


def location_table(G, r):
  c = Chain(G, r)
  return c.L, c.rho, c


def simulate(G, r, u, runs, rng):
  N, out, _ = as_digraph(G)
  counts = np.zeros(N)
  fixed = 0
  edges = [(x, y, 1.0 / len(out[x])) for x in range(N) for y in out[x]]
  for _ in range(runs):
    m = np.zeros(N, bool)
    m[u] = True
    k = 1
    lastwild = -1
    while 0 < k < N:
      ev = []
      w = []
      for x, y, p in edges:
        if m[x] != m[y]:
          ev.append((x, y))
          w.append(r * p if m[x] else p)
      if not ev:
        break
      w = np.asarray(w)
      i = rng.choice(len(ev), p=w / w.sum())
      x, y = ev[i]
      if m[x]:
        if k == N - 1:
          lastwild = y
        m[y] = True
        k += 1
      else:
        m[y] = False
        k -= 1
    if k == N:
      fixed += 1
      counts[lastwild] += 1
  return fixed / runs, counts / max(fixed, 1), fixed


def conditioned(c):
  '''Transposed one-step matrix of the chain conditioned on fixation (Doob transform by h), Moran-step clock.'''
  n = 2**c.N
  h = c.h.copy()
  h[c.full] = 1.0
  ok = h > 1e-14
  R = c.R.tocoo()
  keep = ok[R.row] & ok[R.col]
  rows, cols = R.row[keep], R.col[keep]
  vals = R.data[keep] / c.W[rows] * h[cols] / h[rows]
  stay = np.zeros(n)
  stay[ok] = 1.0
  np.add.at(stay, rows, -vals)
  idx = np.flatnonzero(ok)
  P = sp.csr_matrix(
    (np.concatenate([vals, stay[ok]]), (np.concatenate([rows, idx]), np.concatenate([cols, idx]))), shape=(n, n)
  )
  return P.T.tocsr()


def location(c):
  '''Fixation location distribution for a uniform initial vertex, conditioned on fixation.'''
  w = c.rho / c.rho.sum()
  P = w @ np.clip(c.L, 0, None)
  return P / P.sum()
