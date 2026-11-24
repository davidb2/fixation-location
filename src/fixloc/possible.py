'''Possible fixation locations by reachability among mutant sets.

Vertex v is a possible fixation location from the initial vertex u if and only if the mutant set V minus v is reachable
from the mutant set {u} by steps of positive probability. At finite r the steps are the gain and the loss of one mutant.
Under colonization (r to infinity) only gains occur.
'''
import random

import networkx as nx


def reachable(G, u, finite):
  '''Mutant sets, as bitmasks, reachable from {u} on an undirected graph G with vertices 0..N-1.'''
  N = G.number_of_nodes()
  nb = [list(G.neighbors(x)) for x in range(N)]
  full = (1 << N) - 1
  seen = {1 << u}
  stack = [1 << u]
  while stack:
    X = stack.pop()
    if X == full:
      continue
    for x in range(N):
      if not (X >> x) & 1:
        continue
      for y in nb[x]:
        if (X >> y) & 1:
          continue
        for Y in [X | (1 << y)] + ([X & ~(1 << x)] if finite else []):
          if Y and Y not in seen:
            seen.add(Y)
            stack.append(Y)
  return seen


def kind(G):
  if nx.is_tree(G) and max(d for _, d in G.degree()) <= 2:
    return 'path'
  if nx.is_tree(G) and sorted(d for _, d in G.degree())[-2] == 1:
    return 'star'
  return 'other tree' if nx.is_tree(G) else 'has a cycle'


def check_undirected(sizes=(5, 6, 7)):
  '''Graphs with an impossible fixation location at finite r, and the colonization theorem on every graph and initial
  vertex, for every connected undirected graph of the networkx atlas with the given numbers of vertices.'''
  graphs = [
    nx.convert_node_labels_to_integers(g)
    for g in nx.graph_atlas_g()
    if g.number_of_nodes() in sizes and nx.is_connected(g)
  ]
  finite, colon_ok, colon_bad = {}, 0, []
  for G in graphs:
    N = G.number_of_nodes()
    full = (1 << N) - 1
    cut = set(nx.articulation_points(G))
    imp_f = False
    for u in range(N):
      Sf = reachable(G, u, True)
      Sc = reachable(G, u, False)
      poss_f = {v for v in range(N) if (full ^ (1 << v)) in Sf}
      poss_c = {v for v in range(N) if (full ^ (1 << v)) in Sc}
      if len(poss_f) < N:
        imp_f = True
      if poss_c == set(range(N)) - cut - {u}:
        colon_ok += 1
      else:
        colon_bad.append((nx.to_graph6_bytes(G, header=False).decode().strip(), u))
    if imp_f:
      finite[kind(G)] = finite.get(kind(G), 0) + 1
  tot = {k: sum(kind(G) == k for G in graphs) for k in ('path', 'star', 'other tree', 'has a cycle')}
  return dict(
    graphs=len(graphs),
    finite_r_graphs_with_impossible=finite,
    totals=tot,
    colonization_pairs_matching_theorem=colon_ok,
    colonization_mismatches=colon_bad,
  )


def colonization_last(N, out, u):
  '''Possible fixation locations of the colonization process, by exhaustive search over mutant sets.'''
  seen = set()
  stack = [frozenset([u])]
  last = set()
  while stack:
    X = stack.pop()
    if X in seen:
      continue
    seen.add(X)
    for x in X:
      for y in out[x]:
        if y not in X:
          Y = X | {y}
          if len(Y) == N:
            last.add(y)
          else:
            stack.append(Y)
  return last


def colonization_criterion(N, out, u):
  '''Possible fixation locations of the colonization process from the proposition: t is not u, every other vertex is
  reachable from u in G minus t, and t has an incoming edge from another vertex.'''
  res = set()
  for t in range(N):
    if t == u:
      continue
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from((a, b) for a in range(N) for b in out[a] if a != t and b != t)
    reach = nx.descendants(G, u) | {u}
    has_in = any(t in out[a] for a in range(N) if a != t)
    if reach == set(range(N)) - {t} and has_in:
      res.add(t)
  return res


def check_directed(seed=1, sizes=(3, 4), samples=2000):
  '''The colonization proposition against exhaustive search on every directed graph with the given numbers of vertices,
  from every initial vertex, and on random directed graphs with 5 to 7 vertices.'''
  random.seed(seed)
  checked = bad = 0
  for N in sizes:
    pairs = [(a, b) for a in range(N) for b in range(N) if a != b]
    for mask in range(1 << len(pairs)):
      out = {a: [] for a in range(N)}
      for k, (a, b) in enumerate(pairs):
        if mask >> k & 1:
          out[a].append(b)
      for u in range(N):
        checked += 1
        if colonization_last(N, out, u) != colonization_criterion(N, out, u):
          bad += 1
  for _ in range(samples):
    N = random.choice((5, 6, 7))
    p = random.uniform(0.15, 0.6)
    out = {a: [b for b in range(N) if b != a and random.random() < p] for a in range(N)}
    u = random.randrange(N)
    checked += 1
    if colonization_last(N, out, u) != colonization_criterion(N, out, u):
      bad += 1
  return dict(sizes=list(sizes), random_graphs=samples, seed=seed, cases_checked=checked, mismatches=bad)
