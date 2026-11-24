'''The pair of structures of Figure 4e to g and the structures with one edge added, removed, or moved.

pair_search: every connected graph with N = 5 to 7 vertices against every connected graph with one edge added, removed,
or moved, not isomorphic, with fixation probabilities and fixation times within 1 percent at r = 2. The pairs are ranked
by the Chernoff information of the fixation location, the rate of decrease of the probability of a wrong choice.
records and summarize: the 87 alternatives to the theorized structure of Figure 4. For a set M of two or three monitored
vertices the observation of one fixation event is the vertex of M last occupied by a wild type.
'''
import itertools
from multiprocessing import Pool

import networkx as nx
import numpy as np

from fixloc.exact import Chain, conditioned, location
from fixloc.structure import last_occupied, time_pmf

R = 2.0
N = 7
H = [(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (4, 5), (5, 6)]
G_FIG4 = [(0, 1), (0, 4), (1, 4), (2, 3), (3, 5), (4, 5), (5, 6)]
PAIRS = list(itertools.combinations(range(N), 2))
TRIPLES = list(itertools.combinations(range(N), 3))


def graph(n, edges):
  G = nx.Graph()
  G.add_nodes_from(range(n))
  G.add_edges_from(map(tuple, edges))
  return G


def summ(n, edges, r=R):
  '''Fixation probability, fixation time, and fixation location distribution, initial vertex uniformly random.'''
  c = Chain(graph(n, edges), r)
  w = c.rho / c.rho.sum()
  P = w @ np.clip(c.L, 0, None)
  return float(c.rho.mean()), float((w * c.Tfix_moran).sum()), P / P.sum()


def chernoff(P, Q):
  P = np.asarray(P) + 1e-300
  Q = np.asarray(Q) + 1e-300
  lam = np.linspace(0.01, 0.99, 99)
  return float(max(-np.log((P[:, None] ** lam * Q[:, None] ** (1 - lam)).sum(0))))


def coarse(P, S):
  '''Fixation location at a monitored vertex of S, or elsewhere.'''
  return np.r_[P[list(S)], max(0.0, 1 - P[list(S)].sum())]


def pair_candidates(args):
  '''Pairs of the graph with one edge added, removed, or moved, within 1 percent in fixation probability and time.'''
  n, edges = args
  out = []
  E = set(map(tuple, edges))
  nonE = [e for e in itertools.combinations(range(n), 2) if e not in E]
  rG, TG, PG = summ(n, sorted(E))
  cands = [('remove', E - {e}) for e in E] + [('add', E | {f}) for f in nonE]
  cands += [('move', (E - {e}) | {f}) for e in E for f in nonE]
  G1 = graph(n, E)
  for kind, Hs in cands:
    G2 = graph(n, Hs)
    if not nx.is_connected(G2) or nx.is_isomorphic(G1, G2):
      continue
    rH, TH, PH = summ(n, sorted(Hs))
    if abs(rH - rG) / rG > 0.01 or abs(TH - TG) / TG > 0.01:
      continue
    best = {}
    for k in (1, 2, 3):
      cs = [(chernoff(coarse(PG, s), coarse(PH, s)), s) for s in itertools.combinations(range(n), k)]
      b = max(cs, key=lambda z: z[0])
      best[str(k)] = dict(C=b[0], S=list(b[1]))
    out.append(
      dict(
        N=n,
        kind=kind,
        G=sorted(E),
        H=sorted(Hs),
        rho=[rG, rH],
        T=[TG, TH],
        PG=PG.tolist(),
        PH=PH.tolist(),
        tv=float(0.5 * np.abs(PG - PH).sum()),
        C_full=chernoff(PG, PH),
        best=best,
        edges=[len(E), len(Hs)],
      )
    )
  return out


def time_dist(n, edges, tmax=3000):
  c = Chain(graph(n, edges), R)
  return time_pmf(c, conditioned(c), tmax)


def pair_search(nproc=32):
  '''All pairs, the 30 pairs with the largest Chernoff information, and the 20 best pairs for each N.'''
  jobs = []
  for g in nx.graph_atlas_g():
    n = g.number_of_nodes()
    if 5 <= n <= 7 and nx.is_connected(g):
      jobs.append((n, [tuple(sorted(e)) for e in g.edges()]))
  with Pool(nproc) as pool:
    res = [x for part in pool.map(pair_candidates, jobs, chunksize=4) for x in part]
  res.sort(key=lambda x: -x['C_full'])
  top = res[:30]
  perN = {n: sorted([x for x in res if x['N'] == n], key=lambda x: -x['C_full'])[:20] for n in (5, 6, 7)}
  for x in top + [x for lst in perN.values() for x in lst]:
    if 'C_time' not in x:
      x['C_time'] = chernoff(time_dist(x['N'], x['G']), time_dist(x['N'], x['H']))
  return dict(r=R, n_pairs=len(res), top=top, perN={str(k): v for k, v in perN.items()})


def alternatives():
  '''Connected graphs on the vertices of Figure 4 with one edge of the theorized structure added, removed, or moved.'''
  Hs = set(map(frozenset, H))
  non = [frozenset(e) for e in itertools.combinations(range(N), 2) if frozenset(e) not in Hs]
  seen = {}
  for E in [Hs - {e} for e in Hs] + [Hs | {f} for f in non] + [(Hs - {e}) | {f} for e in Hs for f in non]:
    edges = sorted(tuple(sorted(e)) for e in E)
    if nx.is_connected(graph(N, edges)):
      seen[tuple(edges)] = 1
  return [list(map(list, k)) for k in seen]


def records(edges, tmax=4000):
  '''Exact distributions of the observations of one fixation event on one structure.'''
  c = Chain(graph(N, edges), R)
  PT = conditioned(c)
  w = c.rho / c.rho.sum()
  return dict(
    edges=edges,
    rho=float(c.rho.mean()),
    T=float((w * c.Tfix_moran).sum()),
    L=location(c).tolist(),
    pairs=last_occupied(PT, w, c.full, PAIRS).tolist(),
    triples=last_occupied(PT, w, c.full, TRIPLES).tolist(),
    time=time_pmf(c, PT, tmax).tolist(),
  )


def n95(ps, ph):
  '''Fixation events for probability 0.95 of choosing ps over ph, from the normal approximation 2.706 V / D^2.'''
  ps = np.maximum(np.asarray(ps, float), 1e-15)
  ph = np.maximum(np.asarray(ph, float), 1e-15)
  ps, ph = ps / ps.sum(), ph / ph.sum()
  llr = np.log(ps) - np.log(ph)
  D = float((ps * llr).sum())
  V = float((ps * llr**2).sum() - D**2)
  return np.inf if D <= 1e-12 else max(1.0, float(np.ceil(2.706 * V / D**2)))


def stats(v):
  return dict(
    within_10=float(np.mean(v <= 10)),
    within_100=float(np.mean(v <= 100)),
    within_1000=float(np.mean(v <= 1000)),
    median=float(np.median(v)),
    never=float(np.mean(~np.isfinite(v))),
  )


def summarize(out):
  '''Detection of each alternative by each observation, and distinction between pairs of alternatives.

  out[0] has the records of the theorized structure and out[1:] the records of the alternatives. The outcome of each
  trajectory, fixation or extinction, is the observation with probabilities rho and 1 - rho.'''
  h, alts = out[0], out[1:]

  def det(rec):
    return np.array([n95(rec(a), rec(h)) for a in alts])

  res = dict(alternatives=len(alts))
  for name, sets, key in (('pair', PAIRS, 'pairs'), ('triple', TRIPLES, 'triples')):
    table = np.array([det(lambda s_, i=i: s_[key][i]) for i in range(len(sets))])
    score = [(np.sum(table[i] <= 100), -np.median(table[i])) for i in range(len(sets))]
    order = sorted(range(len(sets)), key=lambda i: score[i], reverse=True)
    res[name] = dict(
      best=list(sets[order[0]]),
      best_stats=stats(table[order[0]]),
      ranking=[[list(sets[i]), int(score[i][0])] for i in order[:6]],
      oracle_stats=stats(table.min(0)),
    )
    if name == 'pair':
      for lab, M in (('pair_3_6', (3, 6)), ('pair_2_3', (2, 3))):
        res[lab] = stats(table[sets.index(M)])
    res[name + '_best_index'] = order[0]
  outcome = lambda s_: [s_['rho'], 1 - s_['rho']]
  res['all_sites'] = stats(det(lambda s_: s_['L']))
  res['times'] = stats(det(lambda s_: s_['time']))
  res['outcome'] = stats(det(outcome))

  def ident(rec):
    Rs = [rec(a) for a in alts]
    ok = [max(n95(Rs[i], Rs[j]), n95(Rs[j], Rs[i])) <= 100 for i, j in itertools.combinations(range(len(alts)), 2)]
    return float(np.mean(ok))

  bp, bt = res['pair_best_index'], res['triple_best_index']
  res['identify'] = dict(
    best_pair=ident(lambda s_: s_['pairs'][bp]),
    best_triple=ident(lambda s_: s_['triples'][bt]),
    all_sites=ident(lambda s_: s_['L']),
    times=ident(lambda s_: s_['time']),
    outcome=ident(outcome),
  )
  g = [a for a in alts if sorted(map(tuple, a['edges'])) == sorted(G_FIG4)][0]
  res['check_fig4_true'] = dict(
    pair_2_3=n95(g['pairs'][PAIRS.index((2, 3))], h['pairs'][PAIRS.index((2, 3))]),
    pair_3_6=n95(g['pairs'][PAIRS.index((3, 6))], h['pairs'][PAIRS.index((3, 6))]),
    all_sites=n95(g['L'], h['L']),
    times=n95(g['time'], h['time']),
  )
  res['rho_theorized'] = h['rho']
  res['rho_alternatives'] = [min(a['rho'] for a in alts), max(a['rho'] for a in alts)]
  return res
