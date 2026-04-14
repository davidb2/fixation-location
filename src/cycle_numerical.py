from itertools import product
import numpy as np

def fix_and_ext(A, x, N, slack, idx, ridx):
  k = 0
  s = +np.inf
  while s >= slack:
    k = k*2 if k > 0 else 1
    b = x@np.linalg.matrix_power(A, k)
    ans = {}
    for idx, p in enumerate(b):
      ans[ridx(idx)] = p
    totalf = 0
    totale = 0
    fix = {}
    ext = {}
    s = 0
    for (i, l), p in ans.items():
      if 0 < l < N:
        s += p
        continue
      elif l == N:
        fix[i] = p
        totalf += p
      elif l == 0:
        ext[i] = p
        totale += p

    if totalf == 0 or totale == 0:
      s = +np.inf
      continue

    for i in range(N):
      fix[i] /= totalf
      ext[i] /= totale

  # print(k)
  return fix, ext

def solve_cycle(N: int, r: float = 1, slack: float = 1e-6, directed: bool = True):
  # (i, l) -> (ip, lp)
  A = {}
  idx = lambda a, b: a * (N+1) + b
  ridx = lambda id: (id // (N+1), id % (N+1))

  A = np.zeros((N*(N+1), N*(N+1)))
  for i, ip in product(range(N), repeat=2):
    for l, lp in product(range(N+1), repeat=2):
      row, col = idx(i, l), idx(ip, lp)
      A[row, col] = 0
      if directed:
        if l in (0, N): A[row, col] = (int(lp == l and ip == i))
        elif ip == i:   A[row, col] = (1/ (r+1)) * int(lp == l - 1)
        elif ip == (i+1) % N: A[row, col] = r/(r+1) * int(lp == l + 1)
      else: # if undirected
        if l in (0, N): A[row, col] = (int(lp == l and ip == i))
        elif ip == i:   A[row, col] = (r/(2*(r+1))) * int(lp == l + 1) + (1/(2*(r+1))) * int(lp == l - 1)
        elif ip == (i-1) % N: A[row, col] = (1/(2*(r+1))) * int(lp == l - 1)
        elif ip == (i+1) % N: A[row, col] = r/(2*(r+1)) * int(lp == l + 1)

  x = np.zeros((N*(N+1),))
  x[idx(0, 1)] = 1

  return fix_and_ext(A, x, N, slack, idx, ridx)