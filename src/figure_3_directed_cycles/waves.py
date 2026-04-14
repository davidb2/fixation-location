#!/usr/bin/env python3.11
"""
Author: David Brewster (dbrewster@g.harvard.edu)
Summary: The goal of this simulation code is to determine
which vertex is likely to be the final vertex to be infected
given that fixation occurs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import setup
from ..cycle_numerical import solve_cycle

plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

if __name__ == '__main__':
  setup()
  SLACK = 1e-6
  for R in (1,2,3):
    for N in (5,10,20,30,40):
      print(N)
      fix, ext = solve_cycle(N, r=R, slack=SLACK)
      print(fix)
      data=[(idx+1, p) for idx, p in fix.items()]
      df = pd.DataFrame(columns=["Last vertex", "p"], data=data)
      ax = sns.barplot(
        df,
        x="Last vertex",
        y="p",
        width=1,
        palette='Greens_d',
      )
      # ax.set(xlabel='Location, $i$', ylabel='Probability last is $i$, $p$')
      ax.set(xlabel='', ylabel='')
      ax.set_xticks(range(len(df)))
      ax.set_xticklabels(['1'] + ([''] * (len(df)-2)) + [f'{N}'])
      ys = np.linspace(0, 1, 11, endpoint=True) 
      ax.set_yticks(ys)
      ax.set_yticklabels([''] * len(ys))
      fig = ax.get_figure()
      plt.plot()
      fig.savefig(f'figs/figure_3_directed_cycles/p-vs-i-R-{R}-fixation-N-{N}-slack-{SLACK}.png', dpi=300, bbox_inches="tight")
      plt.clf()
