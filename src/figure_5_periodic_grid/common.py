import networkx as nx
import pandas as pd

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import *
from dataclasses import dataclass

from ..utils import sample, trial_cond_fix_last_vertex

ROUND = lambda steps, precision: (steps // precision) * precision

@dataclass
class GraphGenerator:
  name: str
  generate: Callable[[int], nx.DiGraph]
  layout: Optional[Callable[[nx.DiGraph], dict]] = None

def work(G, N, samples, initial_node_placements, r):
  print(f"{r=}")
  results = []
  for vertex, steps in sample(lambda: trial_cond_fix_last_vertex(G, initial_node_placements or {0}, r, count_steps=True), samples):
    results.append((N, r, vertex, ROUND(steps, 500)))
  return results

def last_vertex(
  Ns: List[int],
  graph_generator: GraphGenerator,
  Rs: Iterable[float],
  initial_node_placements: Optional[Set[int]] = None,
  samples: int = 1000,
  overwrite: bool = True,
  use_existing_file: bool = False,
  thread: bool = True,
) -> pd.DataFrame:
  file_name = f"./data/figure_5_periodic_grid/{graph_generator.name}-estimated-N-vs-ft-{max(Ns)}.pkl"

  if use_existing_file and Path(file_name).exists():
    df = pd.read_pickle(file_name)
  else:
    data = []
    for N in Ns:
      print(f"{N=}")
      G = graph_generator.generate(N)

      if thread:
        with Pool() as p:
          for results in p.map(partial(work, G, N, samples, initial_node_placements), Rs):
            data.extend(results)
      else:
        for results in map(partial(work, G, N, samples, initial_node_placements), Rs):
          data.extend(results)

    df = pd.DataFrame(data, columns=["Population size", "r", "Last vertex", "Steps"])
    if overwrite:
      df.to_pickle(file_name)

  return df


def is_undirected(G: nx.DiGraph):
  return all((v, u) in G.edges() for (u, v) in G.edges())
