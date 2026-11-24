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
