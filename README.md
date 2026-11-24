# Fixation location in structured populations

Code for the paper *Fixation location in structured populations* by David Brewster, Gábor Lippner, Josef Tkadlec, and
Martin A. Nowak ([arXiv:2605.26411](https://arxiv.org/abs/2605.26411)).

## Layout

```
src/
  fixloc/          Python package: exact solver, figure style, and the methods behind every data set
    make/          one module per figure, which computes the data of each panel
  figs/
    fig1/          one folder per figure, with the figure notebook fig1.ipynb and the image fig1.png
      a/           one folder per panel, with the panel notebook a.ipynb, its data.json, and a.png
    ...
lean/              Lean 4 proofs of the formal results of the Supplementary Information (see lean/README.md)
```

Each panel notebook reads `data.json` in its folder and saves the panel as PNG (600 dpi) and PDF.
Each figure notebook places the panels at their positions and adds the panel letters.
Panels from a hand drawing hold the drawing and a one-line `README.md`. Their notebooks place the drawing as in the
paper. Figure 3 is a single drawing with no notebook for its panel.

## Figures

| Figure | Folder | Panels | Data | Notes |
|---|---|---|---|---|
| 1 | `src/figs/fig1` | a to d | `python -m fixloc.make.fig1` | |
| 2 | `src/figs/fig2` | a to e | `python -m fixloc.make.fig2` | |
| 3 | `src/figs/fig3` | a | none | hand drawing |
| 4 | `src/figs/fig4` | a to g | `python -m fixloc.make.fig4` | a to d are a hand drawing |
| 5 | `src/figs/fig5` | a to f | `python -m fixloc.make.fig5` | a is a hand drawing |
| S1 | `src/figs/figS1` | a | `python -m fixloc.make.figS1` | one panel, no letter |
| S2 | `src/figs/figS2` | a to d | `python -m fixloc.make.figS2` | |
| S3 | `src/figs/figS3` | a | `python -m fixloc.make.figS3` | one panel, no letter |

## Running

```
pip install -e src
cd src/figs/fig1/a
jupyter nbconvert --to notebook --execute --inplace a.ipynb
```

The figures need a LaTeX installation with the `newtx` fonts, because the text is set by LaTeX.
To recompute the data of a figure, run its module from `src/`, for example `python -m fixloc.make.fig1`.
The module overwrites `data.json` in every panel folder of the figure.
Some data sets take hours and were computed on a cluster.

## Notation

| Paper | Code | Meaning |
|---|---|---|
| $N$ | `N` | number of vertices |
| $r$ | `r` | fitness of the mutant; the wild type has fitness 1 |
| $\theta_v$ | `Chain.T`, `theta` | temperature of vertex $v$ |
| $\rho(u)$ | `Chain.rho[u]` | fixation probability from initial vertex $u$ |
| $h(S)$ | `Chain.h[S]` | fixation probability from the mutant set $S$, stored as a bitmask |
| $L(v \mid u)$ | `Chain.L[u, v]` | probability that $v$ is the fixation location, from initial vertex $u$, given fixation |
| fixation location probability | `location(c)`, `P_last` | the same for an initial vertex chosen uniformly, given fixation |
| $T$ | `Chain.Tfix_moran`, `T_expected` | expected fixation time in time steps, given fixation |
| $f_t$ | `f` | probability of fixation by time step $t$, given fixation |
| $m_w(t)$ | `m[t, w]` | probability of a mutant at vertex $w$ at time step $t$, given fixation |
| $H$ | `H` | number of head advances on the directed cycle |
| $m_{12}$, $m_{21}$ | `m12`, `m21` | migration probabilities between the two islands |
| $p_1$ | `p1` | probability of a fixation location on island 1 |
| $r^*$ | `r_star` | unbiased fitness |

A time step of the paper is one reproduction event, called a Moran step in the code.
An active step is a time step that changes the set of mutants.

## License

MIT, see `LICENSE`.
