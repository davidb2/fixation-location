'''Figure S2: the fixation location on the directed cycle under constant selection. Run: python -m fixloc.make.figS2'''
import numpy as np

from fixloc.make import dump
from fixloc.walk import corrected_center, drift, exact_summary, mean_closed, var_asym

RS = [1.5, 2.0, 3.0, 5.0]


def compute():
  out = dict(rs=RS)
  Na = 40
  A = {}
  for r in RS:
    s = exact_summary(Na, r)
    assert abs(s['mean'] - mean_closed(Na, r)) < 1e-8 * max(1.0, s['mean'])
    A[str(r)] = dict(
      prob=s['offset_prob'],
      drift=drift(Na, r) % Na,
      center=corrected_center(Na, r) % Na,
      mean=s['mean'],
      sd=float(np.sqrt(s['var'])),
      sd_formula=float(np.sqrt(var_asym(Na, r))),
    )
  out['a'] = dict(N=Na, curves=A)
  rgrid = np.round(np.arange(1.05, 6.0001, 0.05), 4)
  Nb = [10, 20, 40, 80]
  B = {}
  for N in Nb:
    B[str(N)] = [mean_closed(N, r) / (N - 1) for r in rgrid]
    for r in RS:
      s = exact_summary(N, r)
      assert abs(s['mean'] - mean_closed(N, r)) < 1e-8 * s['mean'], (N, r)
  out['b'] = dict(r=rgrid.tolist(), N=Nb, mean_over_Nm1=B, drift=(rgrid / (rgrid - 1)).tolist())
  Nc = list(range(4, 41))
  C = {}
  for r in RS:
    C[str(r)] = dict(
      err_drift=[abs(mean_closed(N, r) - drift(N, r)) for N in Nc],
      err_corrected=[abs(mean_closed(N, r) - corrected_center(N, r)) for N in Nc],
      err_corrected_recursion=[abs(exact_summary(N, r)['mean'] - corrected_center(N, r)) for N in Nc],
    )
  out['c'] = dict(N=Nc, errors=C)
  Nd = list(range(10, 161, 10))
  D = {}
  for r in RS:
    sd = [float(np.sqrt(exact_summary(N, r)['var'])) for N in Nd]
    D[str(r)] = dict(
      sd_over_sqrtN=[x / np.sqrt(N) for x, N in zip(sd, Nd)],
      drift_level=float(np.sqrt(r * (r + 1) / (r - 1) ** 3)),
      corrected=[float(np.sqrt(max(var_asym(N, r), 0.0)) / np.sqrt(N)) for N in Nd],
    )
  out['d'] = dict(N=Nd, spread=D)
  return out


def write(res):
  rs = res['rs']
  dump('figS2', 'a', dict(rs=rs, **res['a']))
  dump('figS2', 'b', res['b'])
  dump('figS2', 'c', dict(rs=rs, **res['c']))
  dump('figS2', 'd', dict(rs=rs, **res['d']))


if __name__ == '__main__':
  write(compute())
