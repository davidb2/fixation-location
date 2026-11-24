'''Temperature and fixation location of every vertex of every connected undirected graph with N vertices, for N <= 7.'''
import numpy as np
import networkx as nx

from fixloc.exact import Chain


def graphs(N):
  '''Connected undirected graphs with N vertices in the order of the networkx graph atlas.'''
  return [G for G in nx.graph_atlas_g() if G.number_of_nodes() == N and nx.is_connected(G)]


def census(N, r):
  '''Temperature and N times the fixation location probability of every vertex, initial vertex uniformly random.'''
  theta, p = [], []
  for G in graphs(N):
    c = Chain(G, r)
    w = c.rho / c.rho.sum()
    theta += np.asarray(c.T, float).tolist()
    p += (N * (w @ np.clip(c.L, 0, None))).tolist()
  return dict(theta=theta, p_last_N=p)
