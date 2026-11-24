'''A network representing a river with N = 10 habitat patches.

Vertex 0 is the mouth. Vertices 1 and 2 are the stem, with the first split at vertex 2. Vertices 3 and 4 are the left
tributary, with the fork of the left tributary at vertex 4. Vertices 5 and 6 are the left headwaters. Vertex 7 is the fork
of the right tributary, vertex 8 the inner right headwater, and vertex 9 the outer right headwater. Offspring move up or
down the river along bidirectional edges. One directed edge runs from the fork of the left tributary to the inner right
headwater.
'''
import networkx as nx

N = 10
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (2, 7), (7, 8), (7, 9)]
CHANNEL = (4, 8)
MOUTH = 0


def graph(channel=True):
  '''The river as a directed graph, with or without the directed edge.'''
  G = nx.DiGraph()
  G.add_nodes_from(range(N))
  for a, b in EDGES:
    G.add_edge(a, b)
    G.add_edge(b, a)
  if channel:
    G.add_edge(*CHANNEL)
  return G
