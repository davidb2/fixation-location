import networkx as nx
import numpy as np


def undirected_cycle(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for idx in range(N):
    G.add_edge(idx, (idx+1) % N)
    G.add_edge((idx+1) % N, idx)
  return G

def directed_cycle(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for idx in range(N):
    G.add_edge(idx, (idx+1) % N)
  return G


def grid_graph(N: int, periodic=False) -> nx.DiGraph:
  n, m = 0, 0
  for mm in range(int(np.sqrt(N)), 0, -1):
    n, r = divmod(N, mm)
    if r == 0:
      m = mm
      break
  print(m, n)
  def loc(a, b):
    if not periodic and not (0 <= a < m and 0 <= b < n): return None
    if a < 0: a = m-1
    if b < 0: b = n-1
    if a >= m: a = 0
    if b >= n: b = 0
    return n * a + b

  G = nx.DiGraph()
  for i in range(m):
    for j in range(n):
      node = n*i + j
      up = loc(i-1, j)
      down = loc(i+1, j) 
      left = loc(i, j-1) 
      right = loc(i, j+1) 

      if up is not None:
        G.add_edge(node, up)
        G.add_edge(up, node)
      if down is not None:
        G.add_edge(node, down)
        G.add_edge(down, node)
      if left is not None:
        G.add_edge(node, left)
        G.add_edge(left, node)
      if right is not None:
        G.add_edge(node, right)
        G.add_edge(right, node)

  return G