import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from multiprocessing import Pool
from matplotlib.lines import Line2D
from scipy.stats import norm
from ..cycle_numerical import solve_cycle
from ..utils import setup, DoubleLogScale, istarmap

SHOW_NORMAL_APPROX = True
COMPUTE_VARS = False

def do_work(N, R, SLACK):
  fix, ext = solve_cycle(N, r=R, slack=SLACK, directed=False)
  return [(N, R, idx+1, p) for idx, p in fix.items()]

def main3():
  SLACK = 1e-6
  N = 40
  data = []
  with Pool() as pool: 
    for result in tqdm.tqdm(
      pool.istarmap(do_work, [(N, R, SLACK) for R in np.linspace(1, 10, 100+1)]),
      total=100+1,
    ):
      data.extend(result)
  df = pd.DataFrame(columns=["N", "R", "Last vertex", "p"], data=data)
  df.to_csv(f'./data/figure_4_bidirected_cycles/undirected-cycle-fixation-N-{N}-slack-{SLACK}.csv')

def compute_vars():
  SLACK = 1e-6
  last_vertex_df = pd.read_csv(f'./data/figure_4_bidirected_cycles/undirected-cycle-fixation-N-{N}-slack-{SLACK}.csv')
  last_vertex_df = last_vertex_df.drop(columns=["Unnamed: 0"])
  vars = []
  for R in np.linspace(1, 10, 100+1):
    for N in (40,):
      df = last_vertex_df[(last_vertex_df['R'] == R) & (last_vertex_df['N'] == N)]
      mu = ((df['Last vertex']-1)*df['p']).sum()
      var = (((df['Last vertex']-1)-mu)**2 * df['p']).sum()
      vars.append((N, R, mu, var))
  var_df = pd.DataFrame(columns=["N", "R", "mu", "var"], data=vars)
  var_df.to_csv(f'./data/figure_4_bidirected_cycles/undirected-cycle-fixation-variance-N-{N}-slack-{SLACK}.csv')

def main2():
  sns.set_theme(font_scale=2, rc={'text.usetex' : True})
  sns.set_style("ticks")
  SLACK = 1e-6
  N = 40
  last_vertex_df = pd.read_csv(f'./data/figure_4_bidirected_cycles/undirected-cycle-fixation-N-{N}-slack-{SLACK}.csv')
  last_vertex_df = last_vertex_df.drop(columns=["Unnamed: 0"])
  var_df = pd.read_csv(f'./data/figure_4_bidirected_cycles/undirected-cycle-fixation-variance-N-{N}-slack-{SLACK}.csv')
  var_df = var_df.drop(columns=["Unnamed: 0"])

  errors = []
  for R in last_vertex_df['R'].unique():
    lvdf = last_vertex_df[(last_vertex_df['R'] == R) & (last_vertex_df['N'] == N)].drop(columns=["N", "R"])
    lvdf = lvdf.sort_values(by=['Last vertex'])
    ps = lvdf['p'].values

    vdf = var_df[(var_df['R'] == R) & (var_df['N'] == N)].drop(columns=["N", "R"])
    print(vdf)
    mu = vdf['mu'].iloc[0]
    var = vdf['var'].iloc[0]

    x = np.arange(N)
    sigma = np.sqrt(var)
    cdf_values = norm.cdf((x + 0.5 - mu) / sigma)
    cdf_values_left = norm.cdf((x - 0.5 - mu) / sigma)
    approx = cdf_values - cdf_values_left
    error = np.abs(approx - ps).sum() / 2
    errors.append((R, error))

  errors_df = pd.DataFrame(columns=["R", "error"], data=errors)

  g = sns.lineplot(
    var_df,
    x="R",
    y="var",
    marker='o',
    color='blue',
  )
  g.xaxis.set_tick_params(labelbottom=True)
  g.set(xscale='log', yscale='log', xlabel='$r$', ylabel='Variance, $\\sigma^2$')
  # g.figure.savefig(f'./figs/figure_4_bidirected_cycles/variance-N-{N}-slack-{SLACK}.png', dpi=300, bbox_inches="tight")

  ax2 = plt.twinx()
  h = sns.lineplot(
    errors_df,
    x="R",
    y="error",
    marker='X',
    color='red',
    ax=ax2,
  )
  # h.xaxis.set_tick_params(labelbottom=True)
  h.set(xscale='log', yscale='log', xlabel='$r$', ylabel='Total variation distance, $\\Delta$')
  h.legend(handles=[
      Line2D([], [], marker='o', color="blue", label='$\\sigma^2$'),
      Line2D([], [], marker='X', color="red", label='$\\Delta$'),
    ],
  )
  h.figure.savefig(f'./figs/figure_4_bidirected_cycles/var-and-approx-error-N-{N}-slack-{SLACK}.png', dpi=300, bbox_inches="tight")

  plt.show()

def main():
  setup()
  SLACK = 1e-6
  vars = []
  for R in (2,):
    for N in (40,):
      print(N)
      fix, ext = solve_cycle(N, r=R, slack=SLACK, directed=False)
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
      ax.set(xlabel='Location, $i$', ylabel='Probability last is $i$, $p$', yticks=[])
      # ax.set(xlabel='', ylabel='')
      ax.set_xticks(range(len(df)))
      ax.set_xticklabels(['1'] + ([''] * (len(df)-2)) + [f'{N}'])
      # ys = np.linspace(0, 1, 11, endpoint=True) 
      # ax.set_yticks(ys)
      # ax.set_yticklabels([''] * len(ys))
      if SHOW_NORMAL_APPROX:
        x = np.arange(N)
        mu = ((df['Last vertex']-1)*df['p']).sum()
        var = (((df['Last vertex']-1)-mu)**2 * df['p']).sum()
        print(mu, var)
        vars.append((N, R, mu, var))
        sigma = np.sqrt(var)
        cdf_values = norm.cdf((x + 0.5 - mu) / sigma)
        cdf_values_left = norm.cdf((x - 0.5 - mu) / sigma)
        approx = cdf_values - cdf_values_left
        ax.plot(x, approx, marker='o', label='Normal approximation')
        plt.legend()

      sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
      fig = ax.get_figure()
      plt.plot()
      fig.savefig(f'./figs/figure_4_bidirected_cycles/binom-approx-p-vs-i-R-{R}-fixation-N-{N}-slack-{SLACK}.png', dpi=300, bbox_inches="tight")
      plt.clf()

if __name__ == '__main__':
  main2()