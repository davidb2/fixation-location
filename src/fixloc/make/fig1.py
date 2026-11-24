'''Figure 1: a source triangle with one-way links into a sink ladder. Run: python -m fixloc.make.fig1'''
import numpy as np
import networkx as nx

from fixloc.exact import Chain, conditioned, location
from fixloc.make import dump

N = 13
TWO = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10), (10, 11), (11, 12), (4, 9), (6, 11), (7, 12)]
ONE = [(0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (2, 6), (12, 2)]
RS = (1.0, 5.0)


def graph():
  G = nx.DiGraph()
  G.add_nodes_from(range(N))
  for u, v in TWO:
    G.add_edge(u, v)
    G.add_edge(v, u)
  G.add_edges_from(ONE)
  return G


def over_time(c):
  '''f_t and m_w(t) of the chain conditioned on fixation, from a uniform initial vertex weighted by rho.'''
  w = c.rho / c.rho.sum()
  PT = conditioned(c)
  n = 2**N
  member = ((np.arange(n)[:, None] >> np.arange(N)) & 1).astype(float)
  p = np.zeros(n)
  p[1 << np.arange(N)] = w
  T = float((c.rho * c.Tfix_moran).sum() / c.rho.sum())
  f, m, t = [p[c.full]], [p @ member], 0
  while (f[-1] < 0.995 or t < 1.5 * T) and t < 200000:
    p = PT @ p
    t += 1
    f.append(p[c.full])
    m.append(p @ member)
  f, m = np.array(f), np.array(m)
  k = max(1, len(f) // 2000)
  return dict(T_expected=T, t=list(range(0, len(f), k)), f=f[::k].tolist(), m=m[::k].tolist())


def trajectory(w, target, seed=2026, r=1.0):
  '''One simulated fixation trajectory at r = 1 whose last wild type is at the target vertex.'''
  out = {y: sorted(graph().successors(y)) for y in range(N)}
  rng = np.random.default_rng(seed)
  for attempt in range(1, 200001):
    u = int(rng.choice(N, p=w))
    x = np.zeros(N, bool)
    x[u] = True
    hist, t, last = [(0, x.copy())], 0, None
    while 0 < x.sum() < N:
      t += 1
      fit = np.where(x, r, 1.0)
      y = int(rng.choice(N, p=fit / fit.sum()))
      v = out[y][int(rng.integers(len(out[y])))]
      if x[v] != x[y]:
        if x[y] and x.sum() == N - 1 and not x[v]:
          last = v
        x[v] = x[y]
        hist.append((t, x.copy()))
    if x.all() and last == target and len(hist) < 4000:
      break
  counts = [int(s.sum()) for _, s in hist]
  snap = {}
  for k in (3, 7, 10, 12):
    i = next(j for j, n in enumerate(counts) if n >= k)
    snap[str(k)] = [int(hist[i][0]), np.flatnonzero(hist[i][1]).tolist()]
  return dict(attempts=attempt, origin=u, last=last, t_fix=int(hist[-1][0]), snapshots=snap, target=target)


def compute():
  res = {}
  for r in RS:
    c = Chain(graph(), r)
    res[str(r)] = dict(P_last=location(c).tolist(), rho=c.rho.tolist(), theta=np.asarray(c.T, float).tolist(), **over_time(c))
  w = np.array(res['1.0']['rho'])
  res['run'] = trajectory(w / w.sum(), int(np.argmax(res['1.0']['P_last'])))
  res['graph'] = dict(N=N, two_way=TWO, one_way=ONE)
  return res


def write(res):
  g, run = res['graph'], res['run']
  snaps = [v for _, v in sorted(run['snapshots'].items(), key=lambda kv: int(kv[0]))]
  dump('fig1', 'a', dict(**g, origin=run['origin'], snapshots=snaps, t_fix=run['t_fix']))
  dump('fig1', 'b', dict(**g, P_last={r: res[r]['P_last'] for r in ('1.0', '5.0')}))
  for p, r in (('c', '1.0'), ('d', '5.0')):
    x = res[r]
    dump('fig1', p, dict(r=float(r), T_expected=x['T_expected'], t=x['t'], f=x['f'], m=x['m']))


if __name__ == '__main__':
  write(compute())
