import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.colors import LinearSegmentedColormap, FuncNorm
from multiprocessing import Pool
from ..utils import setup

def fix_and_ext_islands(A, x, N1, N2, slack, idx, ridx):
  k = 0
  # slack = 1e-12
  s = +np.inf
  while s >= slack:
    k = k*2 if k > 0 else 1
    b = x@np.linalg.matrix_power(A, k)
    ans = {}
    for idx, p in enumerate(b):
      ans[ridx(idx)] = p
    totalf = 0
    totale = 0
    fix = {}
    ext = {}
    s = 0
    for (n1, n2), p in ans.items():
      if 0 < n1+n2 < N1+N2:
        continue
      elif n1+n2 >= N1+N2:
        last = 0
        if n1+n2 > N1+N2:
          # assert n1+n2 == N1+N2+1, (n1, n2, N1, N2, p)
          # n2 -= 1
          last = 1
        fix[last] = p
        totalf += p
      elif n1+n2 == 0:
        ext[-1] = p
        totale += p

    s = 1-totalf-totale
    #print(totalf + totale, s)
    if totalf == 0: # or totale == 0:
      s = +np.inf
      continue

    # s = 1-totalf-totale
    # print(totalf + totale, s)
    for last in range(2):
      fix[last] /= totalf

  # print(k)
  # print(fix, ext)
  return fix, ext


def solve_islands(N1: int, N2: int, mu12: float, mu21: float, r: float = 1, rho1: float = 1, rho2: float = 1, slack=1e-6):
  # (i, l) -> (ip, lp)
  A = {}
  idx = lambda a, b: a * (N2+1) + b
  ridx = lambda id: (id // (N2+1), (id % (N2+1)) + (N2+1)*int(id == (N1+1)*(N2+1)))

  A = np.zeros(((N1+1)*(N2+1)+1, (N1+1)*(N2+1)+1))
  A[(N1+1)*(N2+1), (N1+1)*(N2+1)] = 1
  inf_limit = np.isinf(r)
  left_total = N1*rho1 + mu12*N2
  right_total = N2*rho2 + mu21*N1

  for n1, n1p in product(range(N1+1), repeat=2):
    for n2, n2p in product(range(N2+1), repeat=2):
      row, col = idx(n1, n2), idx(n1p, n2p)
      A[row, col] = 0
      done = False
      if n1+n2 not in (0, N1+N2) and n1p+n2p in (0, N1+N2):
        # potentially absorbing state.
        done = True
      if n1+n2 in (0, N1+N2):
        A[row, col] = int(n1p == n1 and n2p == n2)
      elif inf_limit:
        k = n1 + n2
        if n1p == n1 and n2p == n2+1:
          # In the r -> infinity birth-death limit, only mutants reproduce.
          A[row, col + done] = (n1/k) * ((N2-n2)*mu12/left_total) + (n2/k) * ((N2-n2)*rho2/right_total)
        elif n1p == n1+1 and n2p == n2:
          A[row, col] = (n1/k) * ((N1-n1)*rho1/left_total) + (n2/k) * ((N1-n1)*mu21/right_total)
        elif n1p == n1 and n2p == n2:
          A[row, col] = (
              (n1/k) * ((n1*rho1 + n2*mu12)/left_total)
            + (n2/k) * ((n2*rho2 + n1*mu21)/right_total)
          )
      else: # r is finite
        w = ((r-1)*n1 + N1) + ((r-1)*n2 + N2)
        if n1p == n1 and n2p == n2+1:
          # picked element in left to reproduce to right
          # picked elemnt in right to reproduce in right
          A[row, col + done] = (n1*r/w) * ((N2-n2)*mu12/left_total) + (n2*r/w) * ((N2-n2)*rho2/right_total)
        elif n1p == n1+1 and n2p == n2:
          # picked element in left to reproduce to left
          # picked elemnt in right to reproduce in left
          A[row, col] = (n1*r/w) * ((N1-n1)*rho1/left_total) + (n2*r/w) * ((N1-n1)*mu21/right_total)
        elif n1p == n1 and n2p == n2-1:
          # picked element in left to reproduce to right
          # picked elemnt in right to reproduce in right
          A[row, col] = ((N1-n1)/w) * (n2*mu12/left_total) + ((N2-n2)/w) * (n2*rho2/right_total)
        elif n1p == n1-1 and n2p == n2:
          # picked element in left to reproduce to left
          # picked elemnt in right to reproduce in left
          A[row, col] = ((N1-n1)/w) * (n1*rho1/left_total) + ((N2-n2)/w) * (n1*mu21/right_total)
        elif n1p == n1 and n2p == n2 and (0 < n1+n2 < N1+N2):
          # picked element in left to reproduce to left
          # picked element in left to reproduce to right
          # picked elemnt in right to reproduce in right
          # picked elemnt in right to reproduce in left
          A[row, col] = (
              ((N1-n1)/w) * (((N1-n1)*rho1 + (N2-n2)*mu12)/left_total)
            + ((N2-n2)/w) * (((N2-n2)*rho2 + (N1-n1)*mu21)/right_total)
            + (n1*r/w) * ((n1*rho1 + n2*mu12)/left_total)
            + (n2*r/w) * ((n2*rho2 + n1*mu21)/right_total)
          )



  x = np.zeros((N1+1)*(N2+1)+1,)
  x[idx(1, 0)] = 1
  return fix_and_ext_islands(A, x, N1, N2, slack, idx, ridx)

def work(N):
  SLACK = 1e-6
  mu12 = 1e-3 / N
  mu21 = 1e-1 / N
  rho1 = (1-mu12) / N
  rho2 = (1-mu21) / N
  print(N)
  N2 = N // 2
  N1 = N-N2
  data = []
  for R in np.linspace(1, 1.5, 20, endpoint=True)[:]:
    fix, ext = solve_islands(N1, N2, mu12=mu12, mu21=mu21, rho1=rho1, rho2=rho2, r=R, slack=SLACK)
    data.append((N, R, fix[0]))
  return data

def work_martin(N, m12, m21):
  SLACK = 1e-6
  N2 = N // 2
  N1 = N-N2
  mu12 = m12 / N
  mu21 = m21 / N
  rho1 = (1-m12) / N
  rho2 = (1-m21) / N 
  print(N, m12, m21)
  data = []
  for R in [np.inf]:#,np.inf]:
    fix, ext = solve_islands(N1, N2, mu12=mu12, mu21=mu21, rho1=rho1, rho2=rho2, r=R, slack=SLACK)
    data.append((N, R, m12, m21, fix[0]))
  return data

def main_martin():
  setup()
  n = 10
  N = 2 * n
  WORK = False
  if WORK:
    ms = np.linspace(1e-4, 1, 100)# np.logspace(-4, 0, 100)
    data = []
    for m12, m21 in itertools.product(ms, repeat=2):
      data.extend(work_martin(N, m12, m21))
    
    df = pd.DataFrame(columns=["N", "r", "m12", "m21", "p1"], data=data)
    df.to_pickle('data/figure_7_islands/island-fixation-david-lol-edited-martin-lin-inf.pkl')
  else:
    df = pd.read_pickle('data/figure_7_islands/island-fixation-david-lol-edited-martin-lin-inf.pkl')

  print(df)

  with sns.plotting_context('poster'), sns.axes_style("ticks"):
    tab10 = sns.color_palette("tab10")
    green = tab10[2]
    purple = tab10[4]

    eps = 0.08
    cmap = LinearSegmentedColormap.from_list(
        "green_white_purple",
        [
            (0.0, purple),
            (0.5, "white"),
            (1.0, green),
        ]
    )

    gamma = 14

    def forward(x):
        x = np.asarray(x)
        y = np.empty_like(x, dtype=float)

        left = x <= 0.5
        right = x > 0.5

        y[left] = 0.5 * (2 * x[left])**gamma
        y[right] = 1 - 0.5 * (2 * (1 - x[right]))**gamma
        return y

    def inverse(y):
        y = np.asarray(y)
        x = np.empty_like(y, dtype=float)

        left = y <= 0.5
        right = y > 0.5

        x[left] = 0.5 * (2 * y[left])**(1/gamma)
        x[right] = 1 - 0.5 * (2 * (1 - y[right]))**(1/gamma)
        return x

    norm = FuncNorm((forward, inverse), vmin=0, vmax=1)
    ax = sns.heatmap(mat:=df[df.r == np.inf].pivot(index='m21', columns='m12', values='p1'), cmap=cmap, norm=norm)
    ax.set(
      xlabel=r'$m_{12}$',
      ylabel=r'$m_{21}$',
      xticks=[0.5, len(mat.columns) - 0.5],
      xticklabels=[r'$10^{-4}$', r'$1$'],
      yticks=[0.5, len(mat.index) - 0.5],
      yticklabels=[r'$10^{-4}$', r'$1$'],
    )
    ax.tick_params(axis='x', rotation=0)
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.set_label(r"$p_1$")
    cbar.set_ticks([0, 0.5, 1])
    fig = ax.get_figure()
    fig.savefig(f'figs/figure_7_islands/david-edited-martin-lin-inf.eps', dpi=300, bbox_inches="tight")
    plt.clf()


def main():
  setup()
  data = []
  # with Pool() as pool:
  results = list(map(work, 2*(np.arange(10)+1)))
  for result in results:
    data.extend(result)

  df = pd.DataFrame(columns=["N", "r", "p1"], data=data)
  print(df)
  ax = sns.lineplot(
    df,
    x="r",
    y="p1",
    hue="N",
    # width=1,
    marker='o',
    # size=.2,
    linestyle='--',
    # palette='Greens_d',
  )
  df.to_pickle('data/island-fixation-david-lol-edited.pkl')
  ax.set(xlabel='', ylabel='')
  fig = ax.get_figure()
  fig.savefig(f'figs/figure_7_islands/david-edited.png', dpi=300, bbox_inches="tight")
  plt.clf()
 

if __name__ == '__main__':
  main_martin()
