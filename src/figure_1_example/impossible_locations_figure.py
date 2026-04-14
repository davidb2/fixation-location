from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import sparse
from scipy.sparse.linalg import spsolve


FITNESS_R = 1.25
EPS_ZERO = 1e-12


@dataclass(frozen=True)
class SolveResult:
  conditional_probs: dict[int, float]
  fixation_probability: float
  elapsed_seconds: float


def build_example_graph() -> tuple[nx.DiGraph, int]:
  n = 12
  start = 1
  edges = [
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 10),
    (2, 11),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 3),
    (9, 6),
    (10, 5),
    (11, 12),
    (12, 6),
  ]
  g = nx.DiGraph()
  g.add_nodes_from(range(1, n + 1))
  g.add_edges_from(edges)
  return g, start


def reachable_from_start(g: nx.DiGraph, start: int) -> set[int]:
  return set(nx.descendants(g, start)) | {start}


def condition_2a_holds(g: nx.DiGraph, start: int, target: int) -> bool:
  h = g.copy()
  h.remove_node(target)
  if start not in h.nodes: return False
  reachable = set(nx.descendants(h, start)) | {start}
  return (set(g.nodes) - {target}) <= reachable


def has_refillable_path_s_neq_t(g: nx.DiGraph, start: int, target: int) -> bool:
  """Check condition (2b) for s != t using a polynomial-time scan.

  We test whether there exist:
  - a path P: u0 -> ... -> t
  - w,z in Gamma^-(u0)
  satisfying refillability constraints.

  The path-off-`w` constraint is checked by asking whether u0 can reach target
  in G-{w}. If yes, there is a path that avoids w.
  """
  nodes = list(g.nodes)
  indeg_start = g.in_degree(start)

  for u0 in nodes:
    if u0 == target: continue
    in_u0 = list(g.predecessors(u0))
    if not in_u0: continue

    for w in in_u0:
      h_avoid_w = g.copy()
      h_avoid_w.remove_node(w)
      if u0 not in h_avoid_w.nodes or target not in h_avoid_w.nodes: continue
      if not nx.has_path(h_avoid_w, u0, target): continue

      for z in in_u0:
        if z == start and indeg_start == 0: continue

        if z != w: return True

        # z == w: need a path start -> w whose interior avoids u0.
        if start == w: return True  # length-0 path

        h_avoid_u0 = g.copy()
        h_avoid_u0.remove_node(u0)
        if start in h_avoid_u0.nodes and w in h_avoid_u0.nodes:
          if nx.has_path(h_avoid_u0, start, w): return True

  return False


def fixation_possible_detector(g: nx.DiGraph, start: int, target: int) -> bool:
  """Detector based on the characterization for the case s != t."""
  if target == start: return False
  if reachable_from_start(g, start) != set(g.nodes): return False
  if condition_2a_holds(g, start, target): return True
  return has_refillable_path_s_neq_t(g, start, target)


def detector_impossible_locations(g: nx.DiGraph, start: int) -> list[int]:
  return sorted(v for v in g.nodes if not fixation_possible_detector(g, start, v))


def solve_exact_birth_death(g: nx.DiGraph, start: int, r: float) -> SolveResult:
  """Exact conditional distribution P(last wild type = v | fixation) using linear."""
  t0 = time.time()

  nodes = sorted(g.nodes)
  n = len(nodes)
  node_to_bit = {v: i for i, v in enumerate(nodes)}
  out_neighbors = {
    node_to_bit[v]: [node_to_bit[u] for u in g.successors(v)] for v in nodes
  }

  n_transient = (1 << n) - 2  # states 1..2^n-2
  full_mask = (1 << n) - 1

  q_rows: list[int] = []
  q_cols: list[int] = []
  q_vals: list[float] = []
  r_fix = np.zeros((n_transient, n), dtype=float)

  for state in range(1, (1 << n) - 1):
    s_idx = state - 1
    n_mutants = state.bit_count()
    total_fitness = r * n_mutants + (n - n_mutants)

    for b in range(n):
      nbrs = out_neighbors[b]
      if not nbrs: continue

      birth_fitness = r if ((state >> b) & 1) else 1.0
      base_prob = (birth_fitness / total_fitness) / len(nbrs)
      b_is_mut = (state >> b) & 1

      for d in nbrs:
        d_is_mut = (state >> d) & 1

        if b_is_mut == d_is_mut:
          q_rows.append(s_idx)
          q_cols.append(s_idx)
          q_vals.append(base_prob)
          continue

        if b_is_mut and not d_is_mut:
          new_state = state | (1 << d)
          if new_state == full_mask:
            r_fix[s_idx, d] += base_prob
          else:
            q_rows.append(s_idx)
            q_cols.append(new_state - 1)
            q_vals.append(base_prob)
        else:
          new_state = state & ~(1 << d)
          if new_state != 0:
            q_rows.append(s_idx)
            q_cols.append(new_state - 1)
            q_vals.append(base_prob)

  q = sparse.csr_matrix((q_vals, (q_rows, q_cols)), shape=(n_transient, n_transient))
  a = sparse.eye(n_transient, format="csr") - q
  rhs = np.zeros(n_transient, dtype=float)
  rhs[(1 << node_to_bit[start]) - 1] = 1.0

  z = spsolve(a.T.tocsc(), rhs)
  fixation_unconditional = np.array([z @ r_fix[:, i] for i in range(n)], dtype=float)
  rho = float(fixation_unconditional.sum())
  if rho <= 0:
    raise RuntimeError("Fixation probability is zero; conditional distribution undefined.")

  fixation_conditional = fixation_unconditional / rho
  probs = {nodes[i]: float(fixation_conditional[i]) for i in range(n)}
  return SolveResult(probs, rho, time.time() - t0)


def exact_impossible_locations(probs: dict[int, float]) -> list[int]:
  return sorted(v for v, p in probs.items() if p < EPS_ZERO)


def select_impossible_for_panel_c(g: nx.DiGraph, impossible_exact: list[int]) -> list[int]:
  if len(impossible_exact) <= 3: return impossible_exact

  base = sorted(impossible_exact)
  h_base = g.subgraph(base).to_undirected()
  best = base
  best_key = (
    nx.number_connected_components(h_base),
    sum(1 for _, d in h_base.degree() if d == 0),
    len(base),
  )

  for cut in nx.articulation_points(h_base):
    cand = sorted(v for v in base if v != cut)
    if len(cand) < 3: continue
    h = g.subgraph(cand).to_undirected()
    key = (
      nx.number_connected_components(h),
      sum(1 for _, d in h.degree() if d == 0),
      len(cand),
    )
    if key > best_key:
      best = cand
      best_key = key

  return best


def draw_four_panel_figure(
  g: nx.DiGraph,
  start: int,
  impossible_panel_c: list[int],
  probs_exact: dict[int, float],
  out_svg: str,
) -> list[str]:
    sns.set_theme(style="whitegrid", context="talk")

    # Original layout.
    pos = nx.kamada_kawai_layout(g.to_undirected())
    nodes = sorted(g.nodes)
    probs_vec = np.array([probs_exact[v] for v in nodes], dtype=float)
    base_node_size = 1050
    highlight_node_size = 1100
    sqrt_sizes = 450 + 3800 * np.sqrt(np.maximum(probs_vec, 0.0))

    def draw_edges(
      ax,
      node_sizes,
      *,
      emphasize: bool = False,
      on_top: bool = False,
      source_margin: float = 2.0,
      target_margin: float = 2.0,
    ):
      edge_art = nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        nodelist=nodes,
        node_size=node_sizes,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=26 if emphasize else 22,
        width=2.9 if emphasize else 2.2,
        edge_color="#1f2328" if emphasize else "#68727f",
        alpha=1.0 if emphasize else 0.95,
        min_source_margin=source_margin,
        min_target_margin=target_margin,
      )
      if on_top and edge_art is not None:
        if isinstance(edge_art, list):
          for patch in edge_art:
            patch.set_zorder(5)
        else:
          edge_art.set_zorder(5)

    def draw_labels(ax):
      nx.draw_networkx_labels(
        g,
        pos,
        labels={v: str(v) for v in nodes},
        font_size=11,
        font_weight="bold",
        font_color="#1f2328",
        ax=ax,
      )

    def draw_panel(ax, panel: str) -> None:
      if panel == "A":
        nx.draw_networkx_nodes(
          g, pos, nodelist=nodes, node_color="#d6d9de", node_size=base_node_size, ax=ax
        )
        draw_edges(ax, base_node_size, on_top=True)
        draw_labels(ax)
        ax.set_title("A. Unweighted Directed Graph", fontweight="bold")

      elif panel == "B":
        colors = ["#d95f02" if v == start else "#9ecae1" for v in nodes]
        nx.draw_networkx_nodes(
          g, pos, nodelist=nodes, node_color=colors, node_size=base_node_size, ax=ax
        )
        draw_edges(ax, base_node_size, on_top=True)
        ax.set_title("B. Initial Mutant At Node 1", fontweight="bold")

      elif panel == "C":
        base_colors = ["#d95f02" if v == start else "#dbe7f3" for v in nodes]
        nx.draw_networkx_nodes(
          g, pos, nodelist=nodes, node_color=base_colors, node_size=base_node_size, ax=ax
        )
        draw_edges(ax, base_node_size, on_top=True)
        if impossible_panel_c:
          nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=impossible_panel_c,
            node_color="#2b7bba",
            node_size=highlight_node_size,
            alpha=0.92,
            ax=ax,
          )
          for v in impossible_panel_c:
            x, y = pos[v]
            ax.text(
              x,
              y,
              "X",
              ha="center",
              va="center",
              color="white",
              fontsize=14,
              fontweight="bold",
            )
        ax.set_title("C. Impossible Fixation Locations", fontweight="bold")

      elif panel == "D":
        cmap = sns.blend_palette(["white", "#e41a1c"], as_cmap=True)
        norm = Normalize(
          vmin=0,
          vmax=float(probs_vec.max()),
        )
        node_colors = [cmap(norm(probs_exact[v])) for v in nodes]
        nx.draw_networkx_nodes(
          g,
          pos,
          nodelist=nodes,
          node_color=node_colors,
          node_size=sqrt_sizes,
          linewidths=1.0,
          edgecolors="#1f2328",
          ax=ax,
        )
        draw_edges(
          ax,
          sqrt_sizes,
          emphasize=False,
          on_top=True,
        )
        ax.set_title("D. Exact P(last wild type = v | fixation)", fontweight="bold")
        cbar = ax.figure.colorbar(
          ScalarMappable(norm=norm, cmap=cmap),
          ax=ax,
          fraction=0.046,
          pad=0.04,
        )
        cbar.set_label("Probability", fontsize=11)

      else:
        raise ValueError(f"Unknown panel '{panel}'")

      ax.set_axis_off()

    ##############
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axs = axes.ravel()
    for ax, panel in zip(axs, ["A", "B", "C", "D"]):
      draw_panel(ax, panel)

    fig.tight_layout()
    fig.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    out_base = Path(out_svg)
    panel_paths: list[str] = []
    for panel in ["D"]: #["A", "B", "C", "D"]:
      panel_fig, panel_ax = plt.subplots(1, 1, figsize=(7, 6))
      draw_panel(panel_ax, panel)
      panel_fig.tight_layout()
      panel_path = out_base.with_name(f"{out_base.stem}_{panel}.svg")
      panel_fig.savefig(panel_path, format="svg", dpi=300, bbox_inches="tight")
      plt.close(panel_fig)
      panel_paths.append(str(panel_path))

    return panel_paths


def main() -> None:
  g, start = build_example_graph()

  solve = solve_exact_birth_death(g, start, r=FITNESS_R)
  impossible_exact = exact_impossible_locations(solve.conditional_probs)
  impossible_panel_c = select_impossible_for_panel_c(g, impossible_exact)
  impossible_detector = detector_impossible_locations(g, start)

  print("Graph summary")
  print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
  print(f"  Weakly connected: {nx.is_weakly_connected(g)}")
  print(f"  Strongly connected: {nx.is_strongly_connected(g)}")
  print(f"  Reachable from start={start}: {reachable_from_start(g, start) == set(g.nodes)}")
  print("")
  print("Process summary")
  print(f"  r = {FITNESS_R}")
  print(f"  Exact fixation probability rho(start={start}) = {solve.fixation_probability:.8f}")
  print(f"  Exact solver runtime: {solve.elapsed_seconds:.3f} sec")
  print(f"  Impossible (detector): {impossible_detector}")
  print(f"  Impossible (exact):    {impossible_exact}")
  print(f"  Impossible (panel C):  {impossible_panel_c}")
  print("")
  print("Exact conditional probabilities:")
  for v in sorted(solve.conditional_probs):
    print(f"  v={v}: {solve.conditional_probs[v]:.8f}")

  out_path = "figs/figure_1_example/impossible_locations_example.svg"
  panel_paths = draw_four_panel_figure(
    g=g,
    start=start,
    impossible_panel_c=impossible_panel_c,
    probs_exact=solve.conditional_probs,
    out_svg=out_path,
  )
  print("")
  print(f"Saved combined SVG figure: {out_path}")
  for panel_path in panel_paths:
    print(f"Saved panel SVG: {panel_path}")


if __name__ == "__main__":
  main()

