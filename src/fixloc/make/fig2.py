'''Figure 2: temperature and fixation location on four structures with N = 10 and on all connected graphs with N = 7. Run: python -m fixloc.make.fig2'''
import numpy as np
import networkx as nx

from fixloc import river
from fixloc.census import census
from fixloc.exact import Chain
from fixloc.make import dump

N = 10
R = 2.0
N_CENSUS = 7


def structures():
  '''Edges of the star, the path, and the irregular graph drawn by networkx with edge probability 0.3 and seed 31.'''
  star = [(0, k) for k in range(1, N)]
  path = [(k, k + 1) for k in range(N - 1)]
  irregular = [tuple(map(int, e)) for e in nx.gnp_random_graph(N, 0.3, seed=31).edges()]
  return dict(star=star, path=path, irregular=irregular)


def summary(edges):
  '''Temperature and fixation location probability, initial vertex uniformly random, conditional on fixation.'''
  G = nx.Graph()
  G.add_nodes_from(range(N))
  G.add_edges_from(edges)
  c = Chain(G, R)
  w = c.rho / c.rho.sum()
  L = np.clip(w @ c.L, 0, None)
  return dict(theta=np.asarray(c.T, float).tolist(), P_last=(L / L.sum()).tolist())


def from_mouth():
  '''Temperature and fixation location probability on the river, initial mutant at the mouth, conditional on fixation.'''
  c = Chain(river.graph(), R)
  L = np.clip(np.asarray(c.L)[river.MOUTH], 0, None)
  return dict(theta=np.asarray(c.T, float).tolist(), P_last=(L / L.sum()).tolist())


def compute():
  res = {k: dict(edges=E, **summary(E)) for k, E in structures().items()}
  res['river'] = dict(edges=river.EDGES, channel=river.CHANNEL, origin=river.MOUTH, **from_mouth())
  res['census'] = census(N_CENSUS, R)
  return res


def write(res):
  for p, k in zip('abcd', ('star', 'path', 'river', 'irregular')):
    dump('fig2', p, dict(r=R, **res[k]))
  e = res['census']
  P = np.array(e['p_last_N'], float) / N_CENSUS
  dump('fig2', 'e', dict(r=R, N=N_CENSUS, theta=e['theta'], P_last=P.tolist()))


if __name__ == '__main__':
  write(compute())
