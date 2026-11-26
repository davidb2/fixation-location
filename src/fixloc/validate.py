'''Checks of the exact solver and of the simulator sweepsim.'''
import networkx as nx
import numpy as np

from fixloc.exact import Chain, conditioned, simulate

D1 = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 6), (6, 4), (1, 6)])


def solver():
  '''The Moran fixation probability on complete graphs, the isothermal theorem on regular graphs, the cycle with five
  vertices at r = 1, and the mean of the fixation time distribution against the escape counts.'''
  res = {}
  for N, r in [(6, 1.0), (6, 2.0), (9, 1.5)]:
    c = Chain(nx.complete_graph(N), r)
    exact = 1 / N if r == 1 else (1 - 1 / r) / (1 - r**-N)
    err = abs(c.rho[0] - exact)
    res[f'K{N}_r{r}'] = dict(rho=c.rho[0], moran_formula=exact, err=err, L_row0=c.L[0].round(6).tolist())
  regular = [('C7', nx.cycle_graph(7)), ('prism', nx.circular_ladder_graph(3))]
  regular.append(('K33', nx.complete_bipartite_graph(3, 3)))
  for name, G in regular:
    c = Chain(G, 2.0)
    exact = (1 - 1 / 2) / (1 - 2.0 ** -G.number_of_nodes())
    res[f'isothermal_{name}'] = dict(rho=c.rho.tolist(), moran=exact, maxerr=float(np.abs(c.rho - exact).max()))
  res['C5_r1_L_row0'] = Chain(nx.cycle_graph(5), 1.0).L[0].round(4).tolist()
  c = Chain(nx.cycle_graph(6), 2.0)
  F = c.time_distribution()
  t = np.arange(1, F.shape[0] + 1)
  mass = F.sum(axis=(0, 2))
  mean_t = (F.sum(axis=2) * t[:, None]).sum(axis=0) / mass
  res['time_dist_C6_r2'] = dict(
    mass=mass.round(8).tolist(),
    rho=c.rho.round(8).tolist(),
    mean_from_distribution=mean_t.round(4).tolist(),
    Tfix_moran=c.Tfix_moran.round(4).tolist(),
  )
  return res


def monte_carlo(runs=40000, seed=1):
  '''Fixation location from initial vertex 0 by the Python simulator, against the exact solver.'''
  rng = np.random.default_rng(seed)
  G1 = nx.connected_watts_strogatz_graph(7, 4, 0.5, seed=3)
  res = {}
  for name, G, r in [('ws7', G1, 2.0), ('ws7_neutral', G1, 1.0), ('digraph7', D1, 1.5)]:
    c = Chain(G, r)
    rho_mc, L_mc, nfix = simulate(G, r, 0, runs, rng)
    se = np.sqrt(L_mc * (1 - L_mc) / max(nfix, 1))
    z = np.abs(L_mc - c.L[0]) / np.where(se > 0, se, 1)
    res[f'mc_{name}'] = dict(
      rho_exact=c.rho[0],
      rho_mc=rho_mc,
      L_exact=c.L[0].round(4).tolist(),
      L_mc=L_mc.round(4).tolist(),
      max_z=float(z.max()),
      fixed_runs=int(nfix),
    )
  return res


def sim_cases():
  from fixloc.monitor import EDGES12

  fig = nx.DiGraph()
  fig.add_nodes_from(range(1, 13))
  fig.add_edges_from(EDGES12)
  return [
    ('small-world 7', nx.connected_watts_strogatz_graph(7, 4, 0.5, seed=3), 2.0, [10, 25, 50]),
    ('digraph 7', D1, 1.5, [10, 30]),
    ('Fig 1 graph (sink)', fig, 1.25, []),
    ('cycle 9', nx.cycle_graph(9), 2.0, [20, 60, 100]),
  ]


def simulator(K=20000, seed=7, cases=None):
  '''z-scores of the simulated fixation probabilities, fixation locations, mean fixation times, and mutants at the
  prescribed times from initial vertex 0, against the exact solver.'''
  from fixloc.sim import run

  res = {}
  for name, G, r, checks in cases or sim_cases():
    R = run(G, r, K, seed, checks, max_runs=K * 2000)
    c = Chain(G, r)
    N = c.N
    ok = R['fixes'] > 0
    rho_hat = R['fixes'] / R['runs']
    z_rho = np.abs(rho_hat - c.rho)[ok] / np.sqrt(c.rho * (1 - c.rho) / R['runs'])[ok]
    Lh = np.zeros((N, N))
    np.add.at(Lh, (R['u'], R['v']), 1)
    Lh = Lh / np.maximum(Lh.sum(axis=1, keepdims=True), 1)
    se = np.sqrt(np.maximum(c.L * (1 - c.L), 1e-12) / np.maximum(R['fixes'][:, None], 1))
    zL = (np.abs(Lh - c.L) / se)[ok]
    Tm = np.array([R['t'][R['u'] == u].mean() if ok[u] else np.nan for u in range(N)])
    Tse = np.array([R['t'][R['u'] == u].std() / np.sqrt(max(R['fixes'][u], 1)) if ok[u] else np.nan for u in range(N)])
    zT = (np.abs(Tm - c.Tfix_moran) / Tse)[ok]
    entry = dict(
      r=r,
      origins_with_takeovers=int(ok.sum()),
      max_z_rho=float(z_rho.max()),
      max_z_location=float(np.nanmax(zL)),
      max_z_duration=float(np.nanmax(zT)),
      cells_checked=int(np.isfinite(zL).sum()),
    )
    if checks:
      M = ((np.arange(2**N)[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
      PT = conditioned(c)
      x = np.zeros(2**N)
      x[1] = 1.0
      ex = {}
      for t in range(max(checks) + 1):
        if t in checks:
          ex[t] = M.T @ x
        x = PT @ x
      sel = R['u'] == 0
      n0 = sel.sum()
      zmax = 0.0
      for k, t in enumerate(checks):
        ph = R['state'][sel, k, :].mean(axis=0)
        s = np.sqrt(np.maximum(ex[t] * (1 - ex[t]), 1e-12) / n0)
        zmax = max(zmax, float((np.abs(ph - ex[t]) / s).max()))
      entry['max_z_node_state_at_checkpoints'] = zmax
    res[name] = entry
  return res
