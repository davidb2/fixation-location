import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from ..graphs import grid_graph


from common import is_undirected, GraphGenerator

ROUND = lambda steps, precision: (steps // precision) * precision


def main_plot_martin():
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
  df = pd.read_csv(f"./data/figure_5_periodic_grid/20260404_164704/simulations.csv")

  G = graph_generator.generate(N)
  if is_undirected(G):
    G = nx.to_undirected(G)
  layout = graph_generator.layout(G)
  init = layout[0]

  fig, ax = plt.subplots(figsize=(15, 10))
  ddf = pd.DataFrame(columns=['x', 'y', 'p', 'r', 'dist'])
  for R in (1,2,5,10,-1): # -1 == inf
    freqs = (
      df[df["r"] == R]
      .groupby("last_vertex")["count"]
      .sum()
      .pipe(lambda s: s / s.sum())
      .to_dict()
    )
    values = [layout[node] + [freqs.get(node, 0)] for node in G.nodes()]
    jp_df = pd.DataFrame(values, columns=['x', 'y', 'p']).sort_values(by=['x', 'y'])
    jp_df['r'] = R
    jp_df['dist'] = np.abs(init[0] - jp_df['x']) + np.abs(init[1] - jp_df['y'])
    ddf = pd.concat([ddf, jp_df], ignore_index=True)

  with sns.plotting_context('poster', font_scale=1.5), sns.axes_style("ticks"):
    g = sns.lineplot(
      data=ddf,
      x='dist',
      y='p',
      marker='o',
      hue='r',
      hue_order=[1,2,5,10,-1],
      palette='tab10',
      ax=ax,
      legend=True,
      errorbar=None,
    )

    handles, labels = g.get_legend_handles_labels()
    g.legend(handles, ['1', '2', '5', '10', '$\infty$'])
    g.set(xlabel='Distance from initial mutant', ylabel='Average probability of being last resident')
    g.tick_params(labelsize=20)
    g.figure.savefig(f'figs/figure_5_periodic_grid/N{N}-p-vs-distance-martin-inf107.eps', dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
  main_plot_martin()