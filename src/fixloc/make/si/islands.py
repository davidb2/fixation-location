'''SI section "Islands". Run: python -m fixloc.make.si.islands (seconds)'''
import numpy as np

from fixloc import islands
from fixloc.make.si import dump

SECTION = 'islands'
RS = np.round(np.linspace(1.0, 1.6, 25), 4)


def last_island(n=10, m12=1e-3, m21=1e-1, rs=(1.0, 1.1, 1.2, 1.26, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0)):
  rows = [islands.last_island(r, n, m12, m21) for r in rs]
  t1, t2 = 1 - m12 + m21, 1 - m21 + m12
  return dict(n=n, N=2 * n, m12=m12, m21=m21, temperature_island1=t1, temperature_island2=t2, rows=rows)


def r_star(Ns=(10, 20, 40, 80), m12=1e-3, m21=1e-1):
  out = {}
  for N in Ns:
    rs, p = islands.r_star(N, m12, m21, RS)
    out[str(N)] = dict(r_star=rs, p1=p.round(4).tolist())
  return dict(m12=m12, m21=m21, r=RS.tolist(), N=out)


if __name__ == '__main__':
  dump(SECTION, 'islands_last_island.json', last_island())
  dump(SECTION, 'r_star.json', r_star())
