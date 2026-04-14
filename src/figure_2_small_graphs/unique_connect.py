import os
import subprocess
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

palette = sns.color_palette(palette='Set1')
palette2 = sns.color_palette(palette='autumn')
COLORS = {
  'start': palette[2],
  'highest': palette[0],
  'possible': palette2[5],
  'not-possible': '#ffffff',
}
N = int(os.environ.get('UNIQUE_CONNECT_N', '5'))
OUTPUT_DIR = Path('figs/figure_2_small_graphs/')
COLOR_GAMMA = 0.6

GRAPH_DISPLAY_ORDER_N4 = [
  'path',
  'star',
  'paw',
  'cycle',
  'diamond',
  'clique',
]

START_CASES_N4 = {
  'path': [0, 1],
  'star': [0, 1],
  'paw': [3, 0, 2],
  'cycle': [0],
  'diamond': [1, 0],
  'clique': [0],
}

FIGURE_BACKGROUND = '#fcfbf7'
PANEL_BACKGROUND = '#f6f3eb'
EDGE_COLOR = '#38424d'
NODE_BASE = np.array([244, 238, 225], dtype=float) / 255.0
NODE_HIGHLIGHT = np.array([223, 177, 47], dtype=float) / 255.0
NODE_EDGE = '#374151'
START_RING = '#1b7f5a'
BEST_RING = '#c44536'


def output_stem(n):
  return f'unique_connected_graphs_n{n}'


def chunk_output_stem(n, chunk_index):
  return f'{output_stem(n)}_part{chunk_index}'


def colorbar_output_stem(n):
  return f'{output_stem(n)}_colorbar'


def connected_graphs_n_labeled(n):
  edges = list(combinations(range(n), 2))
  graphs = []
  for mask in range(1 << len(edges)):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for bit, edge in enumerate(edges):
      if (mask >> bit) & 1:
        graph.add_edge(*edge)
    if nx.is_connected(graph):
      graphs.append((mask, graph))
  return graphs


def graph6_key(graph):
  return nx.to_graph6_bytes(graph, header=False).decode().strip()


def unique_isomorphism_classes(labeled_graphs):
  classes = []
  for mask, graph in labeled_graphs:
    placed = False
    for graph_class in classes:
      if nx.is_isomorphic(graph, graph_class['graph']):
        graph_class['masks'].append(mask)
        placed = True
        break
    if not placed:
      classes.append({
        'rep_mask': mask,
        'graph': graph.copy(),
        'masks': [mask],
      })

  for graph_class in classes:
    graph_class['masks'].sort()

  classes.sort(
    key=lambda item: (
      item['graph'].number_of_edges(),
      tuple(sorted(dict(item['graph'].degree()).values())),
      graph6_key(item['graph']),
    )
  )
  return classes


def start_orbits_under_automorphisms(graph):
  matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph)
  automorphisms = list(matcher.isomorphisms_iter())
  unseen = set(graph.nodes())
  orbits = []

  while unseen:
    node = min(unseen)
    orbit = sorted({mapping[node] for mapping in automorphisms})
    orbits.append(orbit)
    unseen -= set(orbit)

  orbits.sort(key=lambda orbit: (len(orbit), orbit[0]))
  return orbits


def graph_kind_n4(graph):
  degree_sequence = tuple(sorted(dict(graph.degree()).values()))
  edge_count = graph.number_of_edges()
  triangle_count = sum(nx.triangles(graph).values()) // 3

  if edge_count == 3 and degree_sequence == (1, 1, 2, 2):
    return 'path'
  if edge_count == 3 and degree_sequence == (1, 1, 1, 3):
    return 'star'
  if edge_count == 4 and triangle_count == 1:
    return 'paw'
  if edge_count == 4 and degree_sequence == (2, 2, 2, 2):
    return 'cycle'
  if edge_count == 5:
    return 'diamond'
  if edge_count == 6:
    return 'clique'
  raise ValueError(f'Unrecognized N=4 graph with degrees={degree_sequence} and |E|={edge_count}')


def make_template_n4(kind):
  if kind == 'path':
    graph = nx.path_graph(4)
    positions = {
      0: (-1.65, 0.0),
      1: (-0.55, 0.0),
      2: (0.55, 0.0),
      3: (1.65, 0.0),
    }
    return graph, positions

  if kind == 'star':
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (0, 3)])
    positions = {
      0: (0.0, 0.0),
      1: (-1.1, -0.95),
      2: (0.0, 1.15),
      3: (1.1, -0.95),
    }
    return graph, positions

  if kind == 'paw':
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    positions = {
      0: (-0.95, -0.55),
      1: (0.95, -0.55),
      2: (0.0, 0.42),
      3: (0.0, 1.68),
    }
    return graph, positions

  if kind == 'cycle':
    graph = nx.cycle_graph(4)
    positions = {
      0: (-1.05, 0.0),
      1: (0.0, 1.05),
      2: (1.05, 0.0),
      3: (0.0, -1.05),
    }
    return graph, positions

  if kind == 'diamond':
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    positions = {
      0: (-1.0, 0.0),
      1: (0.0, 1.0),
      2: (1.0, 0.0),
      3: (0.0, -1.0),
    }
    return graph, positions

  if kind == 'clique':
    graph = nx.complete_graph(4)
    positions = {
      0: (-0.9, -0.9),
      1: (-0.9, 0.9),
      2: (0.9, 0.9),
      3: (0.9, -0.9),
    }
    return graph, positions

  raise ValueError(f'Unsupported N=4 graph kind: {kind}')


def normalize_positions(raw_positions, target_radius=1.55):
  nodes = sorted(raw_positions)
  coords = np.array([raw_positions[node] for node in nodes], dtype=float)
  coords -= coords.mean(axis=0, keepdims=True)

  span = np.max(np.abs(coords))
  if span > 0:
    coords = coords / span

  return {
    node: tuple((coords[index] * target_radius).tolist())
    for index, node in enumerate(nodes)
  }


def graphviz_positions(graph):
  lines = [
    'graph G {',
    '  graph [layout=neato, overlap=false, splines=false, outputorder=edgesfirst];',
    '  node [shape=circle, width=0.2, height=0.2, fixedsize=true, label=""];',
  ]
  for node in sorted(graph.nodes()):
    lines.append(f'  {node};')
  for u, v in sorted(graph.edges()):
    lines.append(f'  {u} -- {v};')
  lines.append('}')
  graphviz_input = '\n'.join(lines)

  try:
    result = subprocess.run(
      ['neato', '-Tplain'],
      input=graphviz_input,
      text=True,
      capture_output=True,
      check=True,
    )
    raw_positions = {}
    for line in result.stdout.splitlines():
      parts = line.split()
      if parts and parts[0] == 'node':
        raw_positions[int(parts[1])] = (float(parts[2]), float(parts[3]))
    if len(raw_positions) == graph.number_of_nodes():
      return normalize_positions(raw_positions, target_radius=1.9)
  except Exception:
    pass

  fallback = nx.kamada_kawai_layout(graph)
  return normalize_positions(fallback, target_radius=1.9)


def organized_graph_data(graph):
  if graph.number_of_nodes() == 4:
    kind = graph_kind_n4(graph)
    template_graph, template_positions = make_template_n4(kind)
    matcher = nx.algorithms.isomorphism.GraphMatcher(template_graph, graph)
    mapping = next(matcher.isomorphisms_iter())
    positions = {
      graph_node: template_positions[template_node]
      for template_node, graph_node in mapping.items()
    }
    start_cases = [mapping[template_node] for template_node in START_CASES_N4[kind]]
    return positions, start_cases

  positions = graphviz_positions(graph)
  start_cases = [min(orbit) for orbit in start_orbits_under_automorphisms(graph)]
  return positions, start_cases


def fixation_distribution_exact(graph, start):
  # since it is the colonization process, we can use dp.
  # dp over all possible configurations
  # configurations induce a DAG
  n = graph.number_of_nodes()
  full_mask = (1 << n) - 1
  start_mask = 1 << start

  neighbors = [list(graph.neighbors(node)) for node in range(n)]
  degrees = [len(neighbors[node]) for node in range(n)]

  dp = [np.zeros(n, dtype=float) for _ in range(1 << n)]

  # iterate backwards...
  for size in range(n - 1, 0, -1):
    for mask in range(1 << n):
      if mask == full_mask or mask.bit_count() != size: continue

      missing = [node for node in range(n) if not ((mask >> node) & 1)]
      occupied = [node for node in range(n) if (mask >> node) & 1]
      occupied_count = len(occupied)

      add_probability = np.zeros(n, dtype=float)
      for node in occupied:
        weight = 1.0 / occupied_count / degrees[node]
        for neighbor in neighbors[node]:
          if not ((mask >> neighbor) & 1):
            add_probability[neighbor] += weight

      total_add_probability = float(add_probability.sum())
      if total_add_probability <= 0.0: continue

      if size == n - 1:
        terminal = np.zeros(n, dtype=float)
        terminal[missing[0]] = 1.0
        dp[mask] = terminal
        continue

      current = np.zeros(n, dtype=float)
      for node in missing:
        current += add_probability[node] * dp[mask | (1 << node)]
      dp[mask] = current / total_add_probability

  probabilities = dp[start_mask].copy()
  probabilities[start] = 0.0
  return probabilities


def global_probability_max(classes):
  max_probability = 0.0
  for graph_class in classes:
    graph = graph_class['graph']
    _positions, start_cases = organized_graph_data(graph)
    for start in start_cases:
      probabilities = fixation_distribution_exact(graph, start)
      current_max = float(np.max(probabilities))
      if current_max > max_probability:
        max_probability = current_max
  return max_probability


def probability_cmap():
  return mcolors.LinearSegmentedColormap.from_list(
    'unique_connect_probability',
    [
      tuple(NODE_BASE.tolist()),
      tuple(NODE_HIGHLIGHT.tolist()),
    ],
  )


def node_size_for_n(n):
  if n == 4:
    return 840, 1325
  return 215, 350


def edge_width_for_n(n):
  if n == 4:
    return 2.8
  return 3.55


def axis_limits_for_positions(positions, n):
  if n == 4:
    return (-2.2, 2.2), (-1.95, 2.25)

  coords = np.array(list(positions.values()), dtype=float)
  x_center = float((coords[:, 0].min() + coords[:, 0].max()) / 2.0)
  y_center = float((coords[:, 1].min() + coords[:, 1].max()) / 2.0)
  span = float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])))
  half_span = max(1.86, span / 2.0 + 0.58)
  return (
    (x_center - half_span, x_center + half_span),
    (y_center - half_span, y_center + half_span),
  )


def style_axis(ax, n, positions=None):
  ax.set_facecolor(PANEL_BACKGROUND)
  if positions is None:
    if n == 4:
      ax.set_xlim(-2.2, 2.2)
      ax.set_ylim(-1.95, 2.25)
    else:
      ax.set_xlim(-2.3, 2.3)
      ax.set_ylim(-2.3, 2.3)
  else:
    x_limits, y_limits = axis_limits_for_positions(positions, n)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
  ax.set_aspect('equal')
  ax.axis('off')


def probability_node_colors(graph, probabilities, start, norm, cmap):
  candidates = [node for node in graph.nodes() if node != start]
  max_probability = max((float(probabilities[node]) for node in candidates), default=0.0)
  if max_probability > 1e-12:
    best_nodes = {
      node for node in candidates
      if np.isclose(float(probabilities[node]), max_probability, atol=1e-12, rtol=1e-10)
    }

  colors = []
  for node, probability in enumerate(probabilities):
    colors.append(COLORS[
      'start' if node == start else
      'highest' if node in best_nodes else
      'not-possible' if probability <= 1e-12 else 
      'possible'
    ])
  return colors


def draw_structure_panel(ax, graph, positions):
  node_size, _ring_size = node_size_for_n(graph.number_of_nodes())
  style_axis(ax, graph.number_of_nodes(), positions)
  nx.draw_networkx_edges(
    graph,
    positions,
    ax=ax,
    width=edge_width_for_n(graph.number_of_nodes()),
    edge_color=EDGE_COLOR,
  )
  nx.draw_networkx_nodes(
    graph,
    positions,
    ax=ax,
    node_size=node_size,
    node_color=[tuple(NODE_BASE.tolist())] * graph.number_of_nodes(),
    edgecolors=NODE_EDGE,
    linewidths=1.0,
  )


def draw_fixation_panel(ax, graph, positions, start, probabilities, norm, cmap):
  node_size, ring_size = node_size_for_n(graph.number_of_nodes())
  style_axis(ax, graph.number_of_nodes(), positions)
  nx.draw_networkx_edges(
    graph,
    positions,
    ax=ax,
    width=edge_width_for_n(graph.number_of_nodes()),
    edge_color=EDGE_COLOR,
  )
  nx.draw_networkx_nodes(
    graph,
    positions,
    ax=ax,
    node_size=node_size,
    node_color=probability_node_colors(graph, probabilities, start, norm, cmap),
    edgecolors=NODE_EDGE,
    linewidths=1.0,
  )



def ordered_classes(classes, n):
  if n != 4:
    return classes

  class_by_kind = {graph_kind_n4(graph_class['graph']): graph_class for graph_class in classes}
  return [class_by_kind[kind] for kind in GRAPH_DISPLAY_ORDER_N4]


def add_colorbar(fig, axes_for_colorbar, norm, cmap):
  sm = cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  colorbar = fig.colorbar(sm, cax=axes_for_colorbar)
  colorbar.outline.set_visible(False)
  colorbar.ax.tick_params(length=0, labelsize=9, colors=NODE_EDGE)
  tick_values = [0.0, 1 / 7, 0.25, 1 / 3, 0.5, 0.75, 1.0]
  tick_values = [value for value in tick_values if value <= norm.vmax + 1e-12]
  if tick_values[-1] != norm.vmax:
    tick_values.append(norm.vmax)
  colorbar.set_ticks(tick_values)
  return colorbar


def add_colorbar_without_ticks(fig, axes_for_colorbar, norm, cmap):
  sm = cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  colorbar = fig.colorbar(sm, cax=axes_for_colorbar)
  colorbar.outline.set_visible(False)
  colorbar.set_ticks([])
  colorbar.ax.tick_params(length=0, labelleft=False, labelright=False, left=False, right=False)
  colorbar.ax.set_yticks([])
  colorbar.ax.set_yticklabels([])
  colorbar.ax.set_xticks([])
  colorbar.ax.set_xticklabels([])
  return colorbar


def save_row_figure(classes, n):
  ordered = ordered_classes(classes, n)
  probability_max = global_probability_max(ordered)
  cmap = probability_cmap()
  norm = mcolors.PowerNorm(gamma=COLOR_GAMMA, vmin=0.0, vmax=probability_max)

  fig, axes = plt.subplots(
    nrows=len(ordered),
    ncols=5,
    figsize=(14.6, 14.0),
    facecolor=FIGURE_BACKGROUND,
    gridspec_kw={'width_ratios': [1.12, 1.0, 1.0, 1.0, 0.18]},
  )

  for row, graph_class in enumerate(ordered):
    graph = graph_class['graph']
    positions, start_cases = organized_graph_data(graph)
    draw_structure_panel(axes[row, 0], graph, positions)

    for case_index in range(3):
      ax = axes[row, case_index + 1]
      if case_index < len(start_cases):
        start = start_cases[case_index]
        probabilities = fixation_distribution_exact(graph, start)
        draw_fixation_panel(ax, graph, positions, start, probabilities, norm, cmap)
      else:
        style_axis(ax, n)

  colorbar_ax = axes[:, 4]
  for ax in colorbar_ax[:-1]:
    ax.remove()
  cax = colorbar_ax[-1]
  cax.set_facecolor(FIGURE_BACKGROUND)
  add_colorbar(fig, cax, norm, cmap)

  fig.subplots_adjust(left=0.035, right=0.965, top=0.985, bottom=0.025, wspace=0.18, hspace=0.16)

  png_path = OUTPUT_DIR / f'{output_stem(n)}.png'
  eps_path = OUTPUT_DIR / f'{output_stem(n)}.eps'
  fig.savefig(png_path, dpi=400, bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
  fig.savefig(eps_path, format='eps', bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
  plt.close(fig)
  return png_path, eps_path


def save_block_figure(classes, n):
  ordered = ordered_classes(classes, n)
  probability_max = global_probability_max(ordered)
  cmap = probability_cmap()
  norm = mcolors.PowerNorm(gamma=COLOR_GAMMA, vmin=0.0, vmax=probability_max)

  block_cols = 7
  block_rows = 3
  fig = plt.figure(figsize=(23.4, 10.2), facecolor=FIGURE_BACKGROUND)
  outer = fig.add_gridspec(
    block_rows + 1,
    block_cols,
    height_ratios=[1.0, 1.0, 1.0, 0.05],
    left=0.015,
    right=0.985,
    top=0.988,
    bottom=0.052,
    wspace=0.03,
    hspace=0.055,
  )

  for index, graph_class in enumerate(ordered):
    row = index // block_cols
    col = index % block_cols
    graph = graph_class['graph']
    positions, start_cases = organized_graph_data(graph)
    case_count = len(start_cases)

    if case_count == 1:
      block = outer[row, col].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.06,
      )
      anchor_ax = fig.add_subplot(block[0, 0])
      case_axes = [fig.add_subplot(block[0, 1])]
    elif case_count == 2:
      block = outer[row, col].subgridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.055,
      )
      anchor_ax = fig.add_subplot(block[0, 0])
      case_axes = [
        fig.add_subplot(block[0, 1]),
        fig.add_subplot(block[0, 2]),
      ]
    elif case_count == 3:
      block = outer[row, col].subgridspec(
        2,
        2,
        wspace=0.055,
        hspace=0.075,
      )
      anchor_ax = fig.add_subplot(block[0, 0])
      case_axes = [
        fig.add_subplot(block[0, 1]),
        fig.add_subplot(block[1, 0]),
        fig.add_subplot(block[1, 1]),
      ]
    else:
      block = outer[row, col].subgridspec(
        2,
        3,
        width_ratios=[1.05, 1.0, 1.0],
        wspace=0.045,
        hspace=0.07,
      )
      anchor_ax = fig.add_subplot(block[:, 0])
      case_axes = [
        fig.add_subplot(block[0, 1]),
        fig.add_subplot(block[0, 2]),
        fig.add_subplot(block[1, 1]),
        fig.add_subplot(block[1, 2]),
      ]

    draw_structure_panel(anchor_ax, graph, positions)

    for ax, start in zip(case_axes, start_cases):
      probabilities = fixation_distribution_exact(graph, start)
      draw_fixation_panel(ax, graph, positions, start, probabilities, norm, cmap)

  colorbar_ax = fig.add_subplot(outer[block_rows, :])
  colorbar_ax.set_facecolor(FIGURE_BACKGROUND)
  sm = cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  colorbar = fig.colorbar(
    sm,
    cax=colorbar_ax,
    orientation='horizontal',
  )
  colorbar.outline.set_visible(False)
  colorbar.set_ticks([])
  colorbar.ax.tick_params(
    length=0,
    labelbottom=False,
    labeltop=False,
    bottom=False,
    top=False,
  )
  colorbar_ax.set_xticks([])
  colorbar_ax.set_xticklabels([])
  colorbar_ax.set_yticks([])
  colorbar_ax.set_yticklabels([])

  png_path = OUTPUT_DIR / f'{output_stem(n)}.png'
  eps_path = OUTPUT_DIR / f'{output_stem(n)}.eps'
  fig.savefig(png_path, dpi=400, bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
  fig.savefig(eps_path, format='eps', bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
  plt.close(fig)
  return png_path, eps_path


def draw_block_panel(fig, outer, row, col, graph_class, norm, cmap):
  graph = graph_class['graph']
  positions, start_cases = organized_graph_data(graph)
  case_count = len(start_cases)

  if case_count == 1:
    block = outer[row, col].subgridspec(
      1,
      2,
      width_ratios=[1.0, 1.0],
      wspace=0.06,
    )
    anchor_ax = fig.add_subplot(block[0, 0])
    case_axes = [fig.add_subplot(block[0, 1])]
  elif case_count == 2:
    block = outer[row, col].subgridspec(
      1,
      3,
      width_ratios=[1.0, 1.0, 1.0],
      wspace=0.055,
    )
    anchor_ax = fig.add_subplot(block[0, 0])
    case_axes = [
      fig.add_subplot(block[0, 1]),
      fig.add_subplot(block[0, 2]),
    ]
  elif case_count == 3:
    block = outer[row, col].subgridspec(
      2,
      2,
      wspace=0.055,
      hspace=0.075,
    )
    anchor_ax = fig.add_subplot(block[0, 0])
    case_axes = [
      fig.add_subplot(block[0, 1]),
      fig.add_subplot(block[1, 0]),
      fig.add_subplot(block[1, 1]),
    ]
  else:
    block = outer[row, col].subgridspec(
      2,
      3,
      width_ratios=[1.05, 1.0, 1.0],
      wspace=0.045,
      hspace=0.07,
    )
    anchor_ax = fig.add_subplot(block[:, 0])
    case_axes = [
      fig.add_subplot(block[0, 1]),
      fig.add_subplot(block[0, 2]),
      fig.add_subplot(block[1, 1]),
      fig.add_subplot(block[1, 2]),
    ]

  draw_structure_panel(anchor_ax, graph, positions)

  for ax, start in zip(case_axes, start_cases):
    probabilities = fixation_distribution_exact(graph, start)
    draw_fixation_panel(ax, graph, positions, start, probabilities, norm, cmap)


def save_colorbar_figure(n, norm, cmap):
  fig = plt.figure(figsize=(10.5, 0.78), facecolor=FIGURE_BACKGROUND)
  ax = fig.add_axes([0.02, 0.2, 0.96, 0.6])
  ax.set_facecolor(FIGURE_BACKGROUND)

  sm = cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  colorbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
  colorbar.outline.set_visible(False)
  colorbar.set_ticks([])
  colorbar.ax.tick_params(
    length=0,
    labelbottom=False,
    labeltop=False,
    bottom=False,
    top=False,
  )
  ax.set_xticks([])
  ax.set_xticklabels([])
  ax.set_yticks([])
  ax.set_yticklabels([])

  png_path = OUTPUT_DIR / f'{colorbar_output_stem(n)}.png'
  eps_path = OUTPUT_DIR / f'{colorbar_output_stem(n)}.eps'
  fig.savefig(png_path, dpi=400, bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
  fig.savefig(eps_path, format='eps', bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
  plt.close(fig)
  return png_path, eps_path


def save_chunked_block_figures(classes, n):
  ordered = ordered_classes(classes, n)
  probability_max = global_probability_max(ordered)
  cmap = probability_cmap()
  norm = mcolors.PowerNorm(gamma=COLOR_GAMMA, vmin=0.0, vmax=probability_max)
  outputs = []

  chunk_size = 7
  chunk_rows = 4
  chunk_cols = 2
  for chunk_index, start_index in enumerate(range(0, len(ordered), chunk_size), start=1):
    chunk = ordered[start_index:start_index + chunk_size]
    fig = plt.figure(figsize=(10.8, 15.8), facecolor=FIGURE_BACKGROUND)
    outer = fig.add_gridspec(
      chunk_rows,
      chunk_cols,
      left=0.035,
      right=0.965,
      top=0.985,
      bottom=0.03,
      wspace=0.08,
      hspace=0.09,
    )

    for cell_index, graph_class in enumerate(chunk):
      row = cell_index // chunk_cols
      col = cell_index % chunk_cols
      draw_block_panel(fig, outer, row, col, graph_class, norm, cmap)

    png_path = OUTPUT_DIR / f'{chunk_output_stem(n, chunk_index)}.png'
    eps_path = OUTPUT_DIR / f'{chunk_output_stem(n, chunk_index)}.eps'
    fig.savefig(png_path, dpi=400, bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
    fig.savefig(eps_path, format='eps', bbox_inches='tight', facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    outputs.append((png_path, eps_path))

  outputs.append(save_colorbar_figure(n, norm, cmap))
  return outputs


def save_figure(classes, n):
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  if n == 4:
    return [save_row_figure(classes, n)]
  if n == 5:
    return save_chunked_block_figures(classes, n)
  return [save_block_figure(classes, n)]


def main():
  labeled = connected_graphs_n_labeled(N)
  classes = unique_isomorphism_classes(labeled)
  outputs = save_figure(classes, N)

  total_start_cases = sum(len(organized_graph_data(graph_class['graph'])[1]) for graph_class in classes)
  print(f'N = {N}')
  print(f'Connected labeled graphs considered: {len(labeled)}')
  print(f'Unique connected graphs up to isomorphism: {len(classes)}')
  print(f'Unique start-case panels: {total_start_cases}')
  for png_path, eps_path in outputs:
    print(f'Saved PNG: {png_path}')
    print(f'Saved EPS: {eps_path}')


if __name__ == '__main__':
  main()
