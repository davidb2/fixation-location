'''The directed cycle under constant selection as a walk of the block of mutants.

The initial mutant is at vertex 1 and the mutants form a block. At an active step the head of the block advances with
probability p = r/(1+r) and the tail retreats with probability q = 1/(1+r). H is the number of head advances until fixation,
conditional on fixation, and the fixation location is vertex 1 + H modulo N. The exact law of H comes from a recursion over the
block length L and the number K of tail retreats, a[K][L] = p a[K][L-1] + q a[K-1][L+1], solved within each layer K as a
first-order linear filter, with P(fixation after K retreats) = p a[K][N-1].
'''
import numpy as np
from scipy.signal import lfilter


def law_of_K(N, r, tol=1e-17):
  '''Probabilities P(K = k | fixation) for k = 0, 1, ..., and the fixation probability.'''
  p, q = r / (1 + r), 1 / (1 + r)
  cur = np.zeros(N + 1)
  cur[1:N] = p ** np.arange(N - 1)
  out = [p * cur[N - 1]]
  while True:
    b = np.zeros(N + 1)
    b[1 : N - 1] = q * cur[2:N]
    nxt = np.zeros(N + 1)
    nxt[1:N] = lfilter([1.0], [1.0, -p], b[1:N])
    out.append(p * nxt[N - 1])
    cur = nxt
    if cur.sum() < tol:
      break
  P = np.array(out)
  return P / P.sum(), P.sum()


def exact_summary(N, r):
  '''Mean and variance of H, fixation probability, and the probability of each offset of the fixation location from vertex 1.'''
  P, rho = law_of_K(N, r)
  H = np.arange(len(P)) + N - 1
  m = float((P * H).sum())
  v = float((P * (H - m) ** 2).sum())
  loc = np.zeros(N)
  np.add.at(loc, H % N, P)
  return dict(mean=m, var=v, rho=float(rho), offset_prob=loc.tolist())


def mean_closed(N, r):
  '''E[H] from the theorem.'''
  return (N - 1) * r / (r - 1) - (r + 1) / (r - 1) ** 2 + N * (r + 1) / ((r - 1) * (r**N - 1))


def var_asym(N, r):
  '''Var[H] from the theorem without the exponentially small term.'''
  return r * (r + 1) / (r - 1) ** 3 * (N - 2 * (r + 1) / (r - 1))


def drift(N, r):
  '''Drift prediction of E[H].'''
  return (N - 1) * r / (r - 1)


def corrected_center(N, r):
  '''E[H] from the theorem without the exponentially small term.'''
  return (N - 1) * r / (r - 1) - (r + 1) / (r - 1) ** 2


def neutral_deviation(N):
  '''Largest deviation max_k |N p_k - 1| at r = 1, from the first passage law of the unbiased walk of the block length.

  q_j is the probability that the walk from 1 first hits N at step j without hitting 0. A run with j active steps has
  (j + N - 1) / 2 head advances, so p_k = sum of q_j over j = 2(k + iN) + N - 1.'''
  x = np.zeros(N + 1)
  x[1] = 1.0
  q = []
  while True:
    y = np.zeros_like(x)
    y[1 : N - 1] += 0.5 * x[2:N]
    y[2:N] += 0.5 * x[1 : N - 1]
    q.append(0.5 * x[N - 1])
    x = y
    if x.sum() < 1e-16:
      break
  q = np.array(q)
  p = np.zeros(N)
  for k in range(N):
    idx = [2 * (k + j * N) + (N - 1) for j in range(len(q) // (2 * N) + 2)]
    p[k] = N * sum(q[i - 1] for i in idx if 1 <= i <= len(q))
  dev = float(np.abs(N * p - 1).max())
  return dict(sum_p=float(p.sum()), max_abs_N_pk_minus_1=dev, times_sqrtN=dev * np.sqrt(N))
