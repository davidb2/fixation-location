'''Two islands with migration probabilities m12 and m21, under Birth-death updating.

A state counts the mutants on island 1 and on island 2. Fixation through (n - 1, n) has the fixation location on
island 1, and fixation through (n, n - 1) has the fixation location on island 2.
'''
from itertools import product

import numpy as np


def last_island(r, n=10, m12=1e-3, m21=1e-1):
  '''Probability p1 of a fixation location on island 1, by initial island and for a uniform initial vertex.'''
  S = (n + 1) ** 2
  idx = lambda a, b: a * (n + 1) + b
  A = np.zeros((S, S))
  B = np.zeros((S, 2))
  for a in range(n + 1):
    for b in range(n + 1):
      i = idx(a, b)
      if (a, b) in ((0, 0), (n, n)):
        A[i, i] = 1.0
        continue
      W = r * (a + b) + (2 * n - a - b)
      moves = [
        ((a + 1, b), r * a / W * (1 - m12) * (n - a) / n),
        ((a, b + 1), r * a / W * m12 * (n - b) / n),
        ((a, b + 1), r * b / W * (1 - m21) * (n - b) / n),
        ((a + 1, b), r * b / W * m21 * (n - a) / n),
        ((a - 1, b), (n - a) / W * (1 - m12) * a / n),
        ((a, b - 1), (n - a) / W * m12 * b / n),
        ((a, b - 1), (n - b) / W * (1 - m21) * b / n),
        ((a - 1, b), (n - b) / W * m21 * a / n),
      ]
      A[i, i] = sum(p for _, p in moves)
      for (a2, b2), p in moves:
        if p == 0:
          continue
        if (a2, b2) == (n, n):
          B[i, 0 if (a, b) == (n - 1, n) else 1] += p
        elif (a2, b2) != (0, 0):
          A[i, idx(a2, b2)] -= p
  X = np.linalg.solve(A, B)
  s1, s2 = X[idx(1, 0)], X[idx(0, 1)]
  rho1, rho2 = float(s1.sum()), float(s2.sum())
  p1_from1, p1_from2 = float(s1[0] / rho1), float(s2[0] / rho2)
  w1, w2 = n * rho1, n * rho2
  return dict(
    r=r,
    rho_from_island1=rho1,
    rho_from_island2=rho2,
    p1_start_island1=p1_from1,
    p1_start_island2=p1_from2,
    p1_uniform_initial_vertex=(w1 * p1_from1 + w2 * p1_from2) / (w1 + w2),
  )


def fix_and_ext_islands(A, x, N1, N2, slack, ridx):
  '''Absorption from the distribution x, by powers of A doubled until less than slack is unabsorbed.'''
  k = 0
  s = np.inf
  while s >= slack:
    k = k * 2 if k > 0 else 1
    b = x @ np.linalg.matrix_power(A, k)
    ans = {ridx(i): p for i, p in enumerate(b)}
    totalf = 0
    totale = 0
    fix = {}
    ext = {}
    for (n1, n2), p in ans.items():
      if 0 < n1 + n2 < N1 + N2:
        continue
      elif n1 + n2 >= N1 + N2:
        last = 1 if n1 + n2 > N1 + N2 else 0
        fix[last] = p
        totalf += p
      elif n1 + n2 == 0:
        ext[-1] = p
        totale += p
    s = 1 - totalf - totale
    if totalf == 0:
      s = np.inf
      continue
    for last in range(2):
      fix[last] /= totalf
  return fix, ext


def solve_islands(N1, N2, mu12, mu21, r=1, rho1=1, rho2=1, slack=1e-6):
  '''Fixation split by island of the last wild type (key 0 is island 1), from one mutant on island 1.'''
  idx = lambda a, b: a * (N2 + 1) + b
  ridx = lambda i: (i // (N2 + 1), (i % (N2 + 1)) + (N2 + 1) * int(i == (N1 + 1) * (N2 + 1)))
  A = np.zeros(((N1 + 1) * (N2 + 1) + 1, (N1 + 1) * (N2 + 1) + 1))
  A[(N1 + 1) * (N2 + 1), (N1 + 1) * (N2 + 1)] = 1
  inf_limit = np.isinf(r)
  left_total = N1 * rho1 + mu12 * N2
  right_total = N2 * rho2 + mu21 * N1
  for n1, n1p in product(range(N1 + 1), repeat=2):
    for n2, n2p in product(range(N2 + 1), repeat=2):
      row, col = idx(n1, n2), idx(n1p, n2p)
      done = int(n1 + n2 not in (0, N1 + N2) and n1p + n2p in (0, N1 + N2))
      if n1 + n2 in (0, N1 + N2):
        A[row, col] = int(n1p == n1 and n2p == n2)
      elif inf_limit:
        b1, b2 = n1 / (n1 + n2), n2 / (n1 + n2)
        if n1p == n1 and n2p == n2 + 1:
          A[row, col + done] = b1 * ((N2 - n2) * mu12 / left_total) + b2 * ((N2 - n2) * rho2 / right_total)
        elif n1p == n1 + 1 and n2p == n2:
          A[row, col] = b1 * ((N1 - n1) * rho1 / left_total) + b2 * ((N1 - n1) * mu21 / right_total)
        elif n1p == n1 and n2p == n2:
          A[row, col] = b1 * ((n1 * rho1 + n2 * mu12) / left_total) + b2 * ((n2 * rho2 + n1 * mu21) / right_total)
      else:
        w = ((r - 1) * n1 + N1) + ((r - 1) * n2 + N2)
        b1, b2, d1, d2 = n1 * r / w, n2 * r / w, (N1 - n1) / w, (N2 - n2) / w
        if n1p == n1 and n2p == n2 + 1:
          A[row, col + done] = b1 * ((N2 - n2) * mu12 / left_total) + b2 * ((N2 - n2) * rho2 / right_total)
        elif n1p == n1 + 1 and n2p == n2:
          A[row, col] = b1 * ((N1 - n1) * rho1 / left_total) + b2 * ((N1 - n1) * mu21 / right_total)
        elif n1p == n1 and n2p == n2 - 1:
          A[row, col] = d1 * (n2 * mu12 / left_total) + d2 * (n2 * rho2 / right_total)
        elif n1p == n1 - 1 and n2p == n2:
          A[row, col] = d1 * (n1 * rho1 / left_total) + d2 * (n1 * mu21 / right_total)
        elif n1p == n1 and n2p == n2 and 0 < n1 + n2 < N1 + N2:
          A[row, col] = (
            d1 * (((N1 - n1) * rho1 + (N2 - n2) * mu12) / left_total)
            + d2 * (((N2 - n2) * rho2 + (N1 - n1) * mu21) / right_total)
            + b1 * ((n1 * rho1 + n2 * mu12) / left_total)
            + b2 * ((n2 * rho2 + n1 * mu21) / right_total)
          )
  x = np.zeros((N1 + 1) * (N2 + 1) + 1)
  x[idx(1, 0)] = 1
  return fix_and_ext_islands(A, x, N1, N2, slack, ridx)


def p1_migration(N, m12, m21, r):
  '''p1 from one mutant on island 1, for N vertices split into two islands.'''
  N2 = N // 2
  N1 = N - N2
  fix, _ = solve_islands(N1, N2, mu12=m12 / N, mu21=m21 / N, rho1=(1 - m12) / N, rho2=(1 - m21) / N, r=r, slack=1e-6)
  return fix[0]


def p1_grid(r, n=10, ms=np.linspace(1e-4, 1, 100)):
  '''p1 for N = 2n on a grid of migration probabilities, with rows m21 and columns m12.'''
  ms = [float(m) for m in ms]
  return dict(m12=ms, m21=ms, p1=[[p1_migration(2 * n, m12, m21, r) for m12 in ms] for m21 in ms])
