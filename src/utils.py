import networkx as nx
import random
import seaborn as sns
import multiprocessing.pool as mpp

from typing import Optional, Set


def setup():
  sns.set_theme(font_scale=2) #, rc={'text.usetex' : True})
  sns.set_style("ticks", {
    'axes.grid' : False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
  })


def sample(fn, times):
  count = 0
  while count < times:
    if (ans := fn()) is not None:
      yield ans
      count += 1

def trial_cond_fix_last_vertex(G: nx.DiGraph, S: Optional[Set], r: float, count_steps: bool = False):
  if S is None:
    S = {random.choice(list(G.nodes()))}

  N = len(G)
  V = G.nodes()
  mutants = set()
  mutants |= S
  steps = 0

  dier = None
  while V - mutants:
    if not mutants: return None if not count_steps else (None, steps)
    k = len(mutants)
    if random.random() < r*k/(N + (r-1)*k):
      birther = random.choice(list(mutants))
    else:
      birther = random.choice(list(V - mutants))

    dier = random.choice([w for (_, w) in G.out_edges(birther)])
    assert birther != dier
    if birther in mutants:
      mutants.add(dier)
    elif dier in mutants:
      mutants.remove(dier)
    
    steps += 1
  return dier if not count_steps else (dier, steps)


def istarmap(self, func, iterable, chunksize=1):
  """
  starmap-version of imap, source: https://stackoverflow.com/a/57364423
  """
  self._check_running()
  if chunksize < 1:
    raise ValueError(
      "Chunksize must be 1+, not {0:n}".format(
          chunksize))

  task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
  result = mpp.IMapIterator(self)
  self._taskqueue.put((
    self._guarded_task_generation(
      result._job,
      mpp.starmapstar,
      task_batches
    ),
    result._set_length
  ))
  return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

class DoubleLogScale(mscale.ScaleBase):
    """
    A custom y-scale that applies y -> ln( ln(y) ).
    The inverse is Y -> exp( exp(Y) ).
    """
    name = 'doublelog'  # how you'll refer to it, e.g. ax.set_yscale('doublelog')

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        # If you want to parse custom keyword arguments (like 'a' or offsets),
        # you can do that here. Typically you store them as self.some_param.

    def get_transform(self):
        """
        Return the transformation object that does ln(ln(y)).
        """
        return self.DoubleLogTransform()

    def set_default_locators_and_formatters(self, axis):
        """
        Configure tick locators/formatters. We'll just use automatic ones here.
        """
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_minor_locator(ticker.AutoMinorLocator())

    class DoubleLogTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1

        def transform_non_affine(self, values):
            """
            Forward transform: Y = ln( ln(y) ).
            Matplotlib passes in an array of y-values.
            """
            v = np.asanyarray(values)
            # Enforce the domain y > 1:
            if np.any(v <= 1):
                # You could mask or raise an error. 
                # We'll just replace non-valid entries with NaN for demonstration.
                v = np.where(v <= 1, np.nan, v)
            return np.log(np.log(v))

        def inverted(self):
            """
            Return the corresponding inverse transform object.
            """
            return DoubleLogScale.InvertedDoubleLogTransform()

    class InvertedDoubleLogTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1

        def transform_non_affine(self, values):
            """
            Inverse transform: y = exp( exp(Y) ).
            """
            return np.exp(np.exp(values))

        def inverted(self):
            return DoubleLogScale.DoubleLogTransform()


# Register the scale so Matplotlib recognizes 'doublelog'
mscale.register_scale(DoubleLogScale)
