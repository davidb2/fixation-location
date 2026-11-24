'''Figure 5: two islands. Run: python -m fixloc.make.fig5 (about three minutes, for the grids of panel c)'''
import numpy as np

from fixloc import islands, monitor
from fixloc.make import dump

RS = np.round(np.linspace(1.0, 1.5, 51), 4)


def curves(m12=1e-3, m21=1e-1):
  '''p1 against r for N = 2 to 30, from one mutant on island 1.'''
  p1 = {str(2 * n): [islands.last_island(float(r), n, m12, m21)['p1_start_island1'] for r in RS] for n in range(1, 16)}
  return dict(r=RS.tolist(), p1=p1)


def compute():
  grid = {'1': islands.p1_grid(1), 'inf': islands.p1_grid(np.inf)}
  return dict(curves=curves(), grid=grid, watch=monitor.islands_watch())


def write(res):
  cur, w = res['curves'], res['watch']
  dump('fig5', 'b', cur)
  dump('fig5', 'c', res['grid'])
  dump('fig5', 'd', dict(r=cur['r'], p1={N: p for N, p in cur['p1'].items() if int(N) >= 10}))
  exp = {k: w[k]['exp'] for k in ('p_fix_given_mutant_island1', 'p_fix_given_mutant_island2')}
  dump('fig5', 'e', dict(r=w['r'], r_star=w['r_star'], **exp))
  dump('fig5', 'f', {k: w[k] for k in ('r_star', 'quantiles', 'crossing', 'crossing_all_mutant')})


if __name__ == '__main__':
  write(compute())
