import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import *
from dataclasses import dataclass
from graphs import grid_graph

from ..utils import setup
from ..common import is_undirected, GraphGenerator, last_vertex

ROUND = lambda steps, precision: (steps // precision) * precision


def plot_last_vertices_graph(df: pd.DataFrame, N, graph_generator: GraphGenerator, samples, R, relative=True):
  freqs = df[(df["Population size"] == N) & (df["r"] == R)]["Last vertex"].value_counts(normalize=True).to_dict()
  G = graph_generator.generate(N)
  if is_undirected(G):
    G = nx.to_undirected(G)
  cmap = plt.get_cmap('Blues')
  pos = graph_generator.layout(G) if graph_generator.layout else nx.kamada_kawai_layout(G)
  values = [freqs.get(node, 0) for node in G.nodes()]
  norm = plt.Normalize(vmin=min(values) if relative else 0, vmax=max(values) if relative else 1)
  node_colors = [cmap(norm(value)) for value in values]
  print(node_colors)
  
  nx.draw_networkx_nodes(
    G,
    pos,
    cmap=cmap,
    node_color=node_colors,
    edgecolors='black',
  )

  # Create a color bar
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cbar = plt.colorbar(sm, orientation='vertical', ax=plt.gca())
  cbar.set_label('Probability')
  cbar.set_ticks([min(values), max(values)])
  cbar.set_ticklabels([f'min={min(values):.3f}', f'max={max(values):.3f}'])

  nx.draw_networkx_edges(G, pos) # , connectionstyle="arc3,rad=0.1", arrows=True)

  nx.draw_networkx_labels(G, pos, {0: r'$\bigstar$'}, font_color='red')

  plt.savefig(f'figs/figure_5_periodic_grid/{graph_generator.name}-N{N}-R{R}-samples{samples}.png', dpi=300, bbox_inches="tight")
  plt.show()


def main():
  setup()
  N = 121
  Rs = [10]
  SAMPLES = 777
  gen = GraphGenerator(
    name="grid-periodic",
    generate=lambda n: grid_graph(n, periodic=True),
    layout=lambda G: {
      node: [
        ((node // int(np.sqrt(N))) + int(np.sqrt(N)) // 2) % int(np.sqrt(N)),
        ((node % int(np.sqrt(N))) + int(np.sqrt(N)) // 2)  % int(np.sqrt(N)),
      ]
      for node in G.nodes()
    },
  )

  df = last_vertex(
    [N],
    gen,
    samples=SAMPLES,
    Rs=Rs,
  )
  df.to_csv(f'./data/figure_5_periodic_grid/{gen.name}-estimated-N-vs-ft-{N}-samples{SAMPLES}.csv')


def main_old_plot():
  N = 121
  graph_generator = GraphGenerator(
    name="grid-periodic",
    generate=lambda n: grid_graph(n, periodic=True),
    layout=lambda G: {
      node: [
        ((node // int(np.sqrt(N))) + int(np.sqrt(N)) // 2) % int(np.sqrt(N)),
        ((node % int(np.sqrt(N))) + int(np.sqrt(N)) // 2)  % int(np.sqrt(N)),
      ]
      for node in G.nodes()
    },
  )
  df = pd.read_pickle(f"./data/figure_5_periodic_grid/{graph_generator.name}-estimated-N-vs-ft-{N}.pkl")
  plot_last_vertices_graph(df, N, graph_generator, samples=777, R=10)


if __name__ == '__main__':
  setup()
  main()