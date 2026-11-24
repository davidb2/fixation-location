'''Guessing the initial vertex from the fixation location or the fixation time.

The initial vertex is uniformly random. Only trajectories ending in fixation are observed.
Before any observation the initial vertex is u with probability proportional to rho(u). Every score is exact.
'''
import numpy as np
import networkx as nx

from fixloc.exact import Chain


def families(N, seeds=5):
  '''Cycle, path, star, complete graph, 2 x N/2 grid, and seeds instances of six random graph families.'''
  fam = [
    ('cycle', 0, nx.cycle_graph(N)),
    ('path', 0, nx.path_graph(N)),
    ('star', 0, nx.star_graph(N - 1)),
    ('complete', 0, nx.complete_graph(N)),
  ]
  if N in (10, 12, 14):
    fam.append(('grid', 0, nx.grid_2d_graph(2, N // 2)))
  for s in range(seeds):
    fam.append(('random 3-regular', s, nx.random_regular_graph(3, N, seed=100 + s)))
    g = nx.gnp_random_graph(N, 3.5 / (N - 1), seed=200 + s)
    k = 0
    while not nx.is_connected(g):
      k += 1
      g = nx.gnp_random_graph(N, 3.5 / (N - 1), seed=200 + s + 1000 * k)
    fam.append(('Erdos-Renyi', s, g))
    fam.append(('Barabasi-Albert', s, nx.barabasi_albert_graph(N, 2, seed=300 + s)))
    fam.append(('small-world', s, nx.connected_watts_strogatz_graph(N, 4, 0.2, seed=400 + s)))
    fam.append(('random tree', s, nx.random_labeled_tree(N, seed=500 + s)))
    rng = np.random.default_rng(600 + s)
    while True:
      D = nx.DiGraph()
      D.add_nodes_from(range(N))
      for x in range(N):
        for y in rng.choice([z for z in range(N) if z != x], 2, replace=False):
          D.add_edge(x, int(y))
      if nx.is_strongly_connected(D):
        break
    fam.append(('random digraph', s, D))
  return fam


def entropy(p):
  p = p[p > 0]
  return float(-(p * np.log2(p)).sum())


def mutual_info(J):
  '''Mutual information in bits of a joint law J with the initial vertex on axis 0.'''
  pu = J.sum(axis=tuple(range(1, J.ndim)))
  po = J.sum(axis=0)
  return entropy(pu) + entropy(po.ravel()) - entropy(J.ravel())


def map_accuracy(J):
  '''Probability of a correct guess by the likeliest initial vertex given the observation, for J[u, ...].'''
  return float(J.reshape(J.shape[0], -1).max(axis=0).sum())


def scores(args):
  '''Scores of every guess for one graph at fitness r, from args = (family, seed, G, r, with_time).'''
  fam, seed, G, r, with_time = args
  c = Chain(G, r)
  N = c.N
  if c.rho.min() <= 0:
    return None
  J = c.rho[:, None] * c.L
  J = J / J.sum()
  prior = J.sum(axis=1)
  nodes = c.nodes
  d = dict(nx.all_pairs_shortest_path_length(G))
  D = np.array([[d[nodes[u]].get(nodes[v], np.inf) for v in range(N)] for u in range(N)])
  far = 0.0
  for v in range(N):
    col = D[:, v]
    cand = np.where(col == col.max())[0]
    far += J[cand, v].sum() / len(cand)
  rank, cred = 0.0, 0.0
  for v in range(N):
    pv = J[:, v].sum()
    if pv <= 0:
      continue
    post = J[:, v] / pv
    order = np.argsort(-post)
    ranks = np.empty(N)
    ranks[order] = np.arange(1, N + 1)
    rank += (J[:, v] * ranks).sum()
    cs = np.searchsorted(np.cumsum(post[order]), 0.9 - 1e-12) + 1
    cred += pv * cs
  out = dict(
    family=fam,
    seed=seed,
    N=N,
    r=r,
    directed=G.is_directed(),
    chance=1.0 / N,
    prior_best=float(prior.max()),
    acc_location=map_accuracy(J),
    acc_farthest=far,
    mi_location_bits=mutual_info(J),
    max_bits=float(np.log2(N)),
    mean_rank_location=rank,
    cred90_size_location=cred,
    p_origin_is_last=float((J.diagonal()).sum()),
  )
  if with_time:
    F = c.time_distribution(tol=1e-11)
    F = F / F.sum()
    Ft = F.transpose(1, 0, 2)
    Jt = Ft.sum(axis=2)
    out.update(
      acc_duration=map_accuracy(Jt),
      acc_both=map_accuracy(Ft),
      mi_duration_bits=mutual_info(Jt),
      mi_both_bits=mutual_info(Ft),
      time_steps_tracked=int(F.shape[0]),
    )
  return out


def memory(g6s, MI, amp):
  '''Range of the mutual information between the initial vertex and the fixation location over a set of graphs, and its
  rank correlations with features of the graphs.'''
  from scipy.stats import spearmanr

  feat = {k: [] for k in ('diameter', 'mean distance', 'links', 'minimum degree', 'cut vertices')}
  graphs = [nx.from_graph6_bytes(s.encode() if isinstance(s, str) else bytes(s)) for s in g6s]
  for G in graphs:
    d = np.array([k for _, k in G.degree()])
    feat['diameter'].append(nx.diameter(G))
    feat['mean distance'].append(nx.average_shortest_path_length(G))
    feat['links'].append(G.number_of_edges())
    feat['minimum degree'].append(int(d.min()))
    feat['cut vertices'].append(len(list(nx.articulation_points(G))))
  o = dict(
    n_graphs=int(len(MI)),
    MI_min=float(MI.min()),
    MI_median=float(np.median(MI)),
    MI_max=float(MI.max()),
    spearman={k: float(spearmanr(MI, v)[0]) for k, v in feat.items()},
  )
  o['spearman']['amplification factor'] = float(spearmanr(MI, amp)[0])
  G = graphs[int(np.argmax(MI))]
  o['max_graph'] = dict(
    links=G.number_of_edges(), diameter=nx.diameter(G), degrees=sorted((k for _, k in G.degree()), reverse=True)
  )
  o.update(near_complete(graphs, MI))
  return o


def near_complete(graphs, MI):
  '''The five graphs with the least mutual information, and the rank of every graph with at most two edges missing from
  the complete graph, each with the number of missing edges.'''
  n = len(graphs[0])
  miss = np.array([n * (n - 1) // 2 - G.number_of_edges() for G in graphs])
  rank = np.empty(len(MI), int)
  rank[np.argsort(MI)] = np.arange(1, len(MI) + 1)
  near = np.flatnonzero(miss <= 2)
  return dict(
    lowest=[dict(MI=float(MI[i]), missing_links=int(miss[i])) for i in np.argsort(MI)[:5]],
    near_complete=[dict(missing_links=int(miss[i]), MI=float(MI[i]), rank=int(rank[i])) for i in near],
  )
