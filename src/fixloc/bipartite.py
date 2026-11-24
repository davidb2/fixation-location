'''The complete bipartite graph K_{m,n} by lumping, with the star K_{1,n} as a special case.

A state counts the mutants on side A with m vertices and on side B with n vertices. The initial vertex is uniformly
random and every quantity is conditional on fixation.
A vertex of side A has temperature n/m and a vertex of side B has m/n.
'''
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def bipartite(m, n, r):
  '''Amplification, fixation time T and final phase Tend in time steps, Tend from the final phase proposition, the
  probability of a fixation location on side A, and the mean of 1/theta_v at the fixation location.'''
  N = m + n
  S = (m + 1) * (n + 1)
  idx = lambda i, j: i * (n + 1) + j
  rows, cols, vals = [], [], []
  a = np.zeros(S)
  W = np.zeros(S)
  for i in range(m + 1):
    for j in range(n + 1):
      k = idx(i, j)
      W[k] = r * (i + j) + (N - i - j)
      if (i, j) in ((0, 0), (m, n)):
        continue
      moves = (
        (r * i * (n - j) / n, i, j + 1),
        ((m - i) * j / n, i, j - 1),
        (r * j * (m - i) / m, i + 1, j),
        ((n - j) * i / m, i - 1, j),
      )
      for rate, i2, j2 in moves:
        if rate > 0:
          rows.append(k)
          cols.append(idx(i2, j2))
          vals.append(rate)
          a[k] += rate
  R = sp.csr_matrix((vals, (rows, cols)), shape=(S, S))
  inv = np.where(a > 0, 1 / np.where(a > 0, a, 1), 0)
  Q = sp.diags(inv) @ R
  full, zero = idx(m, n), idx(0, 0)
  T = [k for k in range(S) if k not in (full, zero)]
  Qt = Q[T][:, T].tocsc()
  I = sp.identity(len(T), format='csc')
  b = np.asarray(Q[T][:, full].todense()).ravel()
  lu = spla.splu((I - Qt).tocsc())
  h = np.zeros(S)
  h[full] = 1
  h[T] = lu.solve(b)
  mu = np.zeros(S)
  mu[idx(1, 0)] += m / N
  mu[idx(0, 1)] += n / N
  nv = np.zeros(S)
  nv[T] = lu.solve(mu[T], trans='T')
  rho = mu @ h
  hold = np.where(a > 0, W * inv, 0)
  Tm = (nv * h * hold).sum() / rho
  XA, XB = idx(m - 1, n), idx(m, n - 1)
  Tend = (nv[XA] * h[XA] * hold[XA] + nv[XB] * h[XB] * hold[XB]) / rho
  PA = nv[XA] * Q[XA, full] / rho
  PB = nv[XB] * Q[XB, full] / rho
  thA, thB = n / m, m / n
  Tend_id = (r * (N - 1) + 1) / r * (PA * h[XA] / thA + PB * h[XB] / thB)
  wm = (1 - 1 / r) / (1 - r**-N)
  return dict(
    m=m,
    n=n,
    N=N,
    r=r,
    amp=rho / wm,
    rho=rho,
    T=Tm,
    Tend=Tend,
    Tend_id=Tend_id,
    share_end=Tend / Tm,
    P_last_A=PA,
    inv_theta_last=PA / thA + PB / thB,
    h_A=h[XA],
    h_B=h[XB],
  )
