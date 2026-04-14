import argparse
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODE = 'plot'

D = 5
N_MIN = 10
N_MAX = 100
NUM_POINTS = 20
TRIALS = 10_000
CHUNK_SIZE = 250
BASE_SEED = 123
MAX_WORKERS = min(12, os.cpu_count() or 12)
OVERWRITE = False
SHELL_PLOT_NS = None

REQUESTED_DEGREES = [3, 5, 10]
REQUESTED_FIXED_N = 100
REQUESTED_FIXED_N_DEGREES = [3,4,5,6,7,8,9,10,11,12,13,14] + list(range(15, 96, 10))

TRIAL_COLUMNS = [
  'n',
  'trial_id',
  'start_vertex',
  'finish_vertex',
  'start_finish_distance',
  'random_pair_distance',
]

SHELL_COLUMNS = [
  'n',
  'trial_id',
  'distance_k',
  'shell_size',
  'finish_distance',
  'finish_count',
  'avg_fixation_probability',
]

SUMMARY_COLUMNS = [
  'n',
  'metric',
  'metric_label',
  'count',
  'mean_distance',
  'std_distance',
  'se_distance',
  'log10_n',
]

REGRESSION_COLUMNS = [
  'degree',
  'process_intercept',
  'process_intercept_se',
  'baseline_intercept',
  'baseline_intercept_se',
  'shared_slope',
  'shared_slope_se',
  'offset_difference',
  'offset_difference_se',
  'r_squared',
]

FIXED_N_SWEEP_COLUMNS = [
  'degree',
  'n',
  'trial_count',
  'process_mean',
  'process_se',
  'baseline_mean',
  'baseline_se',
  'gap_mean',
  'gap_se',
]

METRIC_STYLES = {
  'start_finish_distance': {
    'label': 'Colonization distance',
    'marker': 'o',
    'linestyle': '-',
    'color': '#1f77b4',
  },
  'random_pair_distance': {
    'label': 'Random-pair distance',
    'marker': 's',
    'linestyle': '--',
    'color': '#ff7f0e',
  },
}


@dataclass(frozen=True)
class RunConfig:
  degree: int
  n_min: int
  n_max: int
  num_points: int
  trials: int
  chunk_size: int = CHUNK_SIZE
  base_seed: int = BASE_SEED
  max_workers: int = MAX_WORKERS
  overwrite: bool = OVERWRITE
  shell_plot_ns: Optional[Sequence[int]] = SHELL_PLOT_NS
  collect_shells: bool = True
  plot_shells: bool = True
  plot_distances: bool = True

  def validate(self):
    if self.degree <= 0:
      raise ValueError(f'Degree must be positive; received {self.degree}.')
    if self.n_min <= 0 or self.n_max <= 0:
      raise ValueError(f'N bounds must be positive; received {self.n_min}, {self.n_max}.')
    if self.n_min > self.n_max:
      raise ValueError(f'n_min must be <= n_max; received {self.n_min} > {self.n_max}.')
    if self.num_points <= 0:
      raise ValueError(f'num_points must be positive; received {self.num_points}.')
    if self.trials <= 0:
      raise ValueError(f'trials must be positive; received {self.trials}.')
    if self.chunk_size <= 0:
      raise ValueError(f'chunk_size must be positive; received {self.chunk_size}.')
    if self.max_workers <= 0:
      raise ValueError(f'max_workers must be positive; received {self.max_workers}.')


def make_run_tag(config):
  return (
    f'd{config.degree}_n{config.n_min}-{config.n_max}_points{config.num_points}'
    f'_trials{config.trials}_seed{config.base_seed}'
  )


def summarize_integer_sequence(values):
  values = [int(value) for value in values]
  if not values:
    raise ValueError('Cannot summarize an empty integer sequence.')
  if len(values) == 1:
    return str(values[0])
  if len(values) >= 3:
    diffs = np.diff(values)
    if np.all(diffs == diffs[0]):
      return f'{values[0]}-{values[-1]}_step{int(diffs[0])}'
  return '-'.join(str(value) for value in values)


def make_run_paths(config):
  tag = make_run_tag(config)
  data_dir = Path('data/figure_6_random_regular')
  figure_dir = Path('figs/figure_6_random_regular')
  return {
    'tag': tag,
    'data_dir': data_dir,
    'figure_dir': figure_dir,
    'trial_csv': data_dir / f'{tag}_trials.csv',
    'shell_csv': data_dir / f'{tag}_shells.csv',
    'summary_csv': data_dir / f'{tag}_summary.csv',
    'regression_csv': data_dir / f'{tag}_shared_slope_fit.csv',
    'distance_plot': figure_dir / f'{tag}_distances_vs_n_logx.png',
  }


def make_degree_sweep_paths(degrees, n_min, n_max, num_points, trials, base_seed):
  degree_tag = f'k{summarize_integer_sequence(degrees)}'
  tag = f'{degree_tag}_n{n_min}-{n_max}_points{num_points}_trials{trials}_seed{base_seed}'
  data_dir = Path('data/figure_6_random_regular')
  figure_dir = Path('figs/figure_6_random_regular')
  return {
    'tag': tag,
    'data_dir': data_dir,
    'figure_dir': figure_dir,
    'offset_csv': data_dir / f'{tag}_shared_slope_offsets.csv',
    'offset_plot': figure_dir / f'{tag}_shared_slope_offsets.png',
  }


def make_fixed_n_sweep_paths(n, degrees, trials, base_seed):
  degree_tag = f'k{summarize_integer_sequence(degrees)}'
  tag = f'n{n}_{degree_tag}_trials{trials}_seed{base_seed}'
  data_dir = Path('data/figure_6_random_regular')
  figure_dir = Path('figs/figure_6_random_regular')
  return {
    'tag': tag,
    'data_dir': data_dir,
    'figure_dir': figure_dir,
    'summary_csv': data_dir / f'{tag}_degree_sweep_summary.csv',
    'distance_plot': figure_dir / f'{tag}_mean_distances_by_degree.png',
    'gap_plot': figure_dir / f'{tag}_distance_gap_by_degree.png',
  }


def ensure_output_dirs(paths):
  paths['data_dir'].mkdir(parents=True, exist_ok=True)
  paths['figure_dir'].mkdir(parents=True, exist_ok=True)


def save_figure(fig, path):
  fig.savefig(path, dpi=300, bbox_inches='tight')
  eps_path = Path(path).with_suffix('.eps')
  fig.savefig(eps_path, format='eps', bbox_inches='tight')
  print(f'Saved figure to {path}')
  print(f'Saved figure to {eps_path}')


def connected_random_regular_graph(degree, n, rng):
  degree = int(degree)
  n = int(n)
  complement_degree = n - 1 - degree
  sample_degree = degree
  use_complement = 0 <= complement_degree < degree
  if use_complement:
    sample_degree = complement_degree
  while True:
    seed = int(rng.randrange(2**32))
    graph = nx.random_regular_graph(sample_degree, n, seed=seed)
    if use_complement:
      graph = nx.complement(graph)
    if nx.is_connected(graph):
      return graph


def run_colonization_process(graph, rng):
  nodes = tuple(graph.nodes())
  neighbors = {node: tuple(graph.neighbors(node)) for node in nodes}
  start = rng.choice(nodes)
  occupied = {start}
  finish = start
  while len(occupied) < len(nodes):
    parent = rng.choice(tuple(occupied))
    child = rng.choice(neighbors[parent])
    if child not in occupied:
      occupied.add(child)
      finish = child
  return start, finish


def log_spaced_valid_ns(degree, n_min, n_max, num_points=30):
  raw = np.logspace(np.log10(int(n_min)), np.log10(int(n_max)), num=int(num_points))
  candidates = np.unique(np.rint(raw).astype(int))
  ns = set()
  for n in candidates:
    n = int(n)
    if n < int(n_min) or n > int(n_max): continue
    if n <= int(degree): continue
    if (n * int(degree)) % 2 != 0:
      if n + 1 <= int(n_max):
        n += 1
      else:
        continue
    if int(n_min) <= n <= int(n_max) and n > int(degree) and (n * int(degree)) % 2 == 0:
      ns.add(n)
  return sorted(ns)


def load_frame(path, columns, defaults=None):
  if not path.exists(): return pd.DataFrame(columns=columns)
  frame = pd.read_csv(path)
  defaults = defaults or {}
  for column in columns:
    if column not in frame.columns:
      frame[column] = defaults.get(column, pd.NA)
  return frame[columns]


def build_summary(trial_df):
  if trial_df.empty: return pd.DataFrame(columns=SUMMARY_COLUMNS)

  long_df = trial_df.melt(
    id_vars=['n', 'trial_id'],
    value_vars=['start_finish_distance', 'random_pair_distance'],
    var_name='metric',
    value_name='distance',
  )
  summary_df = long_df.groupby(['n', 'metric'], as_index=False).agg(
    count=('distance', 'size'),
    mean_distance=('distance', 'mean'),
    std_distance=('distance', 'std'),
  )
  summary_df['std_distance'] = summary_df['std_distance'].fillna(0.0)
  summary_df['se_distance'] = summary_df['std_distance'] / np.sqrt(summary_df['count'])
  summary_df['log10_n'] = np.log10(summary_df['n'].astype(float))
  summary_df['metric_label'] = summary_df['metric'].map(
    {metric: style['label'] for metric, style in METRIC_STYLES.items()}
  )

  return summary_df[SUMMARY_COLUMNS].sort_values(['metric', 'n']).reset_index(drop=True)


def build_fixed_n_degree_row(trial_df, degree, n):
  if trial_df.empty: raise RuntimeError(f'No trial data available for degree={degree}, n={n}.')

  trial_count = int(len(trial_df))
  gap = trial_df['start_finish_distance'].astype(float) - trial_df['random_pair_distance'].astype(float)
  gap_std = float(gap.std(ddof=1)) if trial_count > 1 else 0.0
  gap_se = gap_std / np.sqrt(trial_count) if trial_count > 0 else 0.0
  process_mean = float(trial_df['start_finish_distance'].mean())
  baseline_mean = float(trial_df['random_pair_distance'].mean())
  process_std = float(trial_df['start_finish_distance'].std(ddof=1)) if trial_count > 1 else 0.0
  baseline_std = float(trial_df['random_pair_distance'].std(ddof=1)) if trial_count > 1 else 0.0
  return {
    'degree': int(degree),
    'n': int(n),
    'trial_count': trial_count,
    'process_mean': process_mean,
    'process_se': process_std / np.sqrt(trial_count) if trial_count > 0 else 0.0,
    'baseline_mean': baseline_mean,
    'baseline_se': baseline_std / np.sqrt(trial_count) if trial_count > 0 else 0.0,
    'gap_mean': float(gap.mean()),
    'gap_se': gap_se,
  }


def write_outputs(trial_df, shell_df, summary_df, paths):
  trial_df.sort_values(['n', 'trial_id']).to_csv(paths['trial_csv'], index=False)
  shell_df.sort_values(['n', 'trial_id', 'distance_k']).to_csv(paths['shell_csv'], index=False)
  summary_df.to_csv(paths['summary_csv'], index=False)


def simulate_chunk(args):
  degree, n, chunk_trials, seed, trial_offset, collect_shells = args
  rng = random.Random(int(seed))
  trial_rows = []
  shell_rows = []
  for local_trial_id in range(int(chunk_trials)):
    trial_id = int(trial_offset) + local_trial_id
    graph = connected_random_regular_graph(int(degree), int(n), rng)
    nodes = tuple(graph.nodes())
    start, finish = run_colonization_process(graph, rng)
    distances = nx.single_source_shortest_path_length(graph, start)
    finish_distance = int(distances[finish])
    u, v = rng.sample(nodes, 2)
    random_pair_distance = int(nx.shortest_path_length(graph, u, v))
    trial_rows.append({
      'n': int(n),
      'trial_id': int(trial_id),
      'start_vertex': int(start),
      'finish_vertex': int(finish),
      'start_finish_distance': int(finish_distance),
      'random_pair_distance': int(random_pair_distance),
    })
    if collect_shells:
      shell_sizes = Counter(int(distance) for distance in distances.values())
      for distance_k in sorted(shell_sizes):
        shell_size = int(shell_sizes[distance_k])
        finish_count = int(distance_k == finish_distance)
        shell_rows.append({
          'n': int(n),
          'trial_id': int(trial_id),
          'distance_k': int(distance_k),
          'shell_size': int(shell_size),
          'finish_distance': int(finish_distance),
          'finish_count': int(finish_count),
          'avg_fixation_probability': float(finish_count / shell_size),
        })
  return trial_rows, shell_rows


def process_pool_available():
  try:
    os.sysconf('SC_SEM_NSEMS_MAX')
  except (AttributeError, OSError, PermissionError, ValueError):
    return False

  return True


def map_simulation_chunks(chunk_args, max_workers):
  if max_workers <= 1 or not process_pool_available():
    if max_workers > 1:
      print('ProcessPoolExecutor unavailable in this environment; falling back to serial execution.')
    return [simulate_chunk(chunk_arg) for chunk_arg in chunk_args]
  try:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      return list(executor.map(simulate_chunk, chunk_args))
  except PermissionError as error:
    print(f'ProcessPoolExecutor unavailable ({error}); falling back to serial execution.')
    return [simulate_chunk(chunk_arg) for chunk_arg in chunk_args]


def simulate_for_n(config, n):
  chunk_args = []
  remaining = int(config.trials)
  chunk_index = 0
  trial_offset = 0
  while remaining > 0:
    chunk_trials = min(int(config.chunk_size), remaining)
    seed = int(config.base_seed) + int(config.degree) * 1_000_000 + int(n) * 10_000 + chunk_index
    chunk_args.append((
      int(config.degree),
      int(n),
      int(chunk_trials),
      int(seed),
      int(trial_offset),
      bool(config.collect_shells),
    ))
    remaining -= chunk_trials
    trial_offset += chunk_trials
    chunk_index += 1
  trial_rows = []
  shell_rows = []
  for chunk_trial_rows, chunk_shell_rows in map_simulation_chunks(chunk_args, config.max_workers):
    trial_rows.extend(chunk_trial_rows)
    shell_rows.extend(chunk_shell_rows)
  trial_df = pd.DataFrame.from_records(trial_rows, columns=TRIAL_COLUMNS)
  shell_df = pd.DataFrame.from_records(shell_rows, columns=SHELL_COLUMNS)
  return trial_df, shell_df


def run_simulations(config, paths):
  ns = log_spaced_valid_ns(config.degree, config.n_min, config.n_max, num_points=config.num_points)
  if not ns:
    raise RuntimeError(
      f'No valid N values produced for degree={config.degree}; '
      f'check the configuration bounds.'
    )

  if config.overwrite:
    trial_df = pd.DataFrame(columns=TRIAL_COLUMNS)
    shell_df = pd.DataFrame(columns=SHELL_COLUMNS)
  else:
    trial_df = load_frame(paths['trial_csv'], TRIAL_COLUMNS)
    shell_df = load_frame(paths['shell_csv'], SHELL_COLUMNS)

  print(f'Using degree={config.degree}, Ns={ns}')
  print(f'Workers={config.max_workers}, trials per N={config.trials}')
  print(f'Writing trial data to {paths["trial_csv"]}')

  if config.collect_shells:
    print(f'Writing shell data to {paths["shell_csv"]}')
  for n in ns:
    existing_trial_count = 0 if trial_df.empty else int((trial_df['n'] == n).sum())
    existing_shell_count = 0 if shell_df.empty else int((shell_df['n'] == n).sum())
    has_complete_trials = existing_trial_count == config.trials
    has_complete_shells = (not config.collect_shells) or existing_shell_count > 0
    if not config.overwrite and has_complete_trials and has_complete_shells:
      print(f'Skipping degree={config.degree}, n={n}: found {existing_trial_count} saved trials')
      continue
    if existing_trial_count > 0:
      trial_df = trial_df[trial_df['n'] != n].copy()
    if existing_shell_count > 0:
      shell_df = shell_df[shell_df['n'] != n].copy()
    new_trial_df, new_shell_df = simulate_for_n(config, n)
    trial_df = pd.concat([trial_df, new_trial_df], ignore_index=True)
    if config.collect_shells:
      shell_df = pd.concat([shell_df, new_shell_df], ignore_index=True)
    summary_df = build_summary(trial_df)
    write_outputs(trial_df, shell_df, summary_df, paths)
    current_summary = summary_df[summary_df['n'] == n].set_index('metric')
    process_mean = current_summary.at['start_finish_distance', 'mean_distance']
    process_se = current_summary.at['start_finish_distance', 'se_distance']
    baseline_mean = current_summary.at['random_pair_distance', 'mean_distance']
    baseline_se = current_summary.at['random_pair_distance', 'se_distance']
    print(
      f'degree={config.degree:3d}, n={n:4d} | '
      f'colonization mean={process_mean:.4f} (SE={process_se:.4f}) | '
      f'random mean={baseline_mean:.4f} (SE={baseline_se:.4f})'
    )
  print(f'Finished simulation data in {paths["data_dir"]}')


def fit_shared_slope_offset(summary_df, degree):
  if summary_df.empty:
    raise RuntimeError(f'No summary data available for degree={degree}.')

  if summary_df['n'].nunique() < 2:
    raise RuntimeError('Shared-slope regression needs at least two distinct N values.')

  metric_df = summary_df[summary_df['metric'].isin(METRIC_STYLES)].copy()
  if metric_df['metric'].nunique() != len(METRIC_STYLES):
    raise RuntimeError('Shared-slope regression needs both distance metrics.')

  metric_df = metric_df.sort_values(['n', 'metric']).reset_index(drop=True)
  x = metric_df['log10_n'].astype(float).to_numpy()
  is_random = (metric_df['metric'] == 'random_pair_distance').astype(float).to_numpy()
  y = metric_df['mean_distance'].astype(float).to_numpy()
  se = metric_df['se_distance'].fillna(0.0).astype(float).to_numpy()

  weights = np.ones_like(y)
  positive_se = se > 0
  weights[positive_se] = 1.0 / np.square(se[positive_se])
  design = np.column_stack([np.ones_like(x), is_random, x])
  weighted_design = design * np.sqrt(weights)[:, None]
  weighted_y = y * np.sqrt(weights)
  coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
  process_intercept, offset_difference, shared_slope = coefficients
  fitted = design @ coefficients
  residuals = y - fitted
  weighted_mean = np.average(y, weights=weights)
  ss_res = float(np.sum(weights * np.square(residuals)))
  ss_tot = float(np.sum(weights * np.square(y - weighted_mean)))
  r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
  xtwx = weighted_design.T @ weighted_design
  dof = max(len(y) - design.shape[1], 1)
  sigma2 = ss_res / dof
  covariance = sigma2 * np.linalg.pinv(xtwx)
  process_intercept_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
  offset_difference_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
  shared_slope_se = float(np.sqrt(max(covariance[2, 2], 0.0)))
  baseline_intercept = float(process_intercept + offset_difference)
  baseline_variance = covariance[0, 0] + covariance[1, 1] + 2.0 * covariance[0, 1]
  baseline_intercept_se = float(np.sqrt(max(baseline_variance, 0.0)))
  regression_df = pd.DataFrame([{
    'degree': int(degree),
    'process_intercept': float(process_intercept),
    'process_intercept_se': process_intercept_se,
    'baseline_intercept': baseline_intercept,
    'baseline_intercept_se': baseline_intercept_se,
    'shared_slope': float(shared_slope),
    'shared_slope_se': shared_slope_se,
    'offset_difference': float(offset_difference),
    'offset_difference_se': offset_difference_se,
    'r_squared': float(r_squared),
  }], columns=REGRESSION_COLUMNS)
  return regression_df


def apply_log_x_axis(ax):
  ax.set_xscale('log')
  ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
  ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
  ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
  ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def plot_distance_summary(summary_df, path, degree, trials, regression_df=None):
  if summary_df.empty: raise RuntimeError('No summary data available to plot.')

  sns.set_theme(style='ticks', context='talk')
  fig, ax = plt.subplots(figsize=(8.5, 5.5))
  for metric, style in METRIC_STYLES.items():
    metric_df = summary_df[summary_df['metric'] == metric].sort_values('n')
    ax.plot(
      metric_df['n'],
      metric_df['mean_distance'],
      marker=style['marker'],
      linestyle=style['linestyle'],
      color=style['color'],
      linewidth=2.2,
      label=style['label'],
    )
    ax.errorbar(
      metric_df['n'],
      metric_df['mean_distance'],
      yerr=metric_df['se_distance'],
      fmt='none',
      color=style['color'],
      capsize=3,
      linewidth=1.2,
    )
    if regression_df is not None and not regression_df.empty and metric_df['n'].nunique() >= 2:
      fit_row = regression_df.iloc[0]
      n_grid = np.logspace(np.log10(metric_df['n'].min()), np.log10(metric_df['n'].max()), 200)
      intercept = fit_row['process_intercept']
      if metric == 'random_pair_distance':
        intercept = fit_row['baseline_intercept']
      ax.plot(
        n_grid,
        intercept + fit_row['shared_slope'] * np.log10(n_grid),
        color=style['color'],
        linestyle=':',
        linewidth=2.0,
        label=f'{style["label"]} (shared-slope fit)',
      )
  apply_log_x_axis(ax)
  ax.set(
    xlabel='N',
    ylabel='Average shortest-path distance',
    title=f'Distances on connected random {degree}-regular graphs (trials={trials})',
  )
  ax.legend(frameon=False)
  sns.despine()
  fig.tight_layout()
  save_figure(fig, path)
  plt.close(fig)


def plot_shared_slope_offsets(offset_df, path, n_min, n_max):
  if offset_df.empty:
    raise RuntimeError('No offset summary data available to plot.')
  sns.set_theme(style='ticks', context='talk')
  fig, ax = plt.subplots(figsize=(8, 5))
  ax.errorbar(
    offset_df['degree'],
    offset_df['offset_difference'],
    yerr=offset_df['offset_difference_se'],
    color='#2ca02c',
    marker='o',
    linestyle='-',
    linewidth=2.0,
    capsize=3,
  )
  ax.set(
    xlabel='Degree k',
    ylabel='Shared-slope offset difference',
    title=f'Random-pair minus colonization offset for N in [{n_min}, {n_max}]',
  )
  sns.despine()
  fig.tight_layout()
  save_figure(fig, path)
  plt.close(fig)


def get_selected_shell_ns(shell_df, shell_plot_ns):
  available_ns = sorted(int(n) for n in shell_df['n'].unique())
  if not available_ns:
    raise RuntimeError('No shell data available to plot.')
  if shell_plot_ns is None:
    return [available_ns[-1]]
  if isinstance(shell_plot_ns, int):
    requested_ns = [int(shell_plot_ns)]
  else:
    requested_ns = [int(n) for n in shell_plot_ns]
  selected_ns = [n for n in requested_ns if n in available_ns]
  if not selected_ns:
    raise RuntimeError(f'No shell data found for shell_plot_ns={requested_ns}.')
  return selected_ns


def make_shell_plot_path(paths, selected_ns):
  if len(selected_ns) == 1:
    suffix = f'n{selected_ns[0]}'
  else:
    suffix = f'n{selected_ns[0]}-{selected_ns[-1]}'
  return paths['figure_dir'] / f'{paths["tag"]}_fixation_probability_by_distance_{suffix}.png'


def plot_shell_probabilities(shell_df, paths, shell_plot_ns):
  selected_ns = get_selected_shell_ns(shell_df, shell_plot_ns)
  plot_df = shell_df[shell_df['n'].isin(selected_ns)].copy()
  plot_df = plot_df[plot_df['distance_k'] > 0].copy()
  if plot_df.empty:
    raise RuntimeError('No shell rows with distance_k > 0 are available to plot.')
  plot_df['n'] = plot_df['n'].astype(int)
  plot_df['distance_k'] = plot_df['distance_k'].astype(int)
  mean_df = plot_df.groupby(['n', 'distance_k'], as_index=False).agg(
    avg_fixation_probability=('avg_fixation_probability', 'mean')
  )
  sns.set_theme(style='ticks', context='talk')
  shell_plot_path = make_shell_plot_path(paths, selected_ns)
  if len(selected_ns) == 1:
    n = selected_ns[0]
    single_mean_df = mean_df[mean_df['n'] == n].sort_values('distance_k')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
      single_mean_df['distance_k'],
      single_mean_df['avg_fixation_probability'],
      color=sns.color_palette('crest', 1)[0],
      marker='o',
      linewidth=2.0,
    )
    ax.set(
      xlabel='Distance from start',
      ylabel='Average fixation probability',
      title=f'Fixation probability by distance from the start (N={n})',
    )
    sns.despine()
    fig.tight_layout()
    save_figure(fig, shell_plot_path)
    plt.close(fig)
  else:
    selected_ns = sorted(selected_ns)
    num_cols = min(4, len(selected_ns))
    num_rows = int(np.ceil(len(selected_ns) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
    flat_axes = axes.flat
    for ax, n in zip(flat_axes, selected_ns):
      current_mean_df = mean_df[mean_df['n'] == n].sort_values('distance_k')
      ax.plot(
        current_mean_df['distance_k'],
        current_mean_df['avg_fixation_probability'],
        color=sns.color_palette('crest', 1)[0],
        marker='o',
        linewidth=2.0,
      )
      ax.set_title(f'N={n}')
      ax.set_xlabel('Distance from start')
      ax.set_ylabel('Average fixation probability')
      sns.despine(ax=ax)
    for ax in list(flat_axes)[len(selected_ns):]:
      ax.remove()
    fig.tight_layout()
    save_figure(fig, shell_plot_path)
    plt.close(fig)


def plot_fixed_n_distances_by_degree(summary_df, path, fixed_n):
  if summary_df.empty:
    raise RuntimeError('No fixed-N degree summary available to plot.')
  sns.set_theme(style='ticks', context='talk')
  fig, ax = plt.subplots(figsize=(8.5, 5.5))
  ax.errorbar(
    summary_df['degree'],
    summary_df['process_mean'],
    yerr=summary_df['process_se'],
    color=METRIC_STYLES['start_finish_distance']['color'],
    marker=METRIC_STYLES['start_finish_distance']['marker'],
    linestyle=METRIC_STYLES['start_finish_distance']['linestyle'],
    linewidth=2.0,
    capsize=3,
    label=METRIC_STYLES['start_finish_distance']['label'],
  )
  ax.errorbar(
    summary_df['degree'],
    summary_df['baseline_mean'],
    yerr=summary_df['baseline_se'],
    color=METRIC_STYLES['random_pair_distance']['color'],
    marker=METRIC_STYLES['random_pair_distance']['marker'],
    linestyle=METRIC_STYLES['random_pair_distance']['linestyle'],
    linewidth=2.0,
    capsize=3,
    label=METRIC_STYLES['random_pair_distance']['label'],
  )
  ax.set(
    xlabel='Degree k',
    ylabel='Average shortest-path distance',
    title=f'Mean distances by degree at N={fixed_n}',
  )
  ax.legend(frameon=False)
  sns.despine()
  fig.tight_layout()
  save_figure(fig, path)
  plt.close(fig)


def plot_fixed_n_gap_by_degree(summary_df, path, fixed_n):
  if summary_df.empty:
    raise RuntimeError('No fixed-N degree summary available to plot.')
  sns.set_theme(style='ticks', context='talk')
  fig, ax = plt.subplots(figsize=(8.5, 5.5))
  ax.errorbar(
    summary_df['degree'],
    summary_df['gap_mean'],
    yerr=summary_df['gap_se'],
    color='#2ca02c',
    marker='o',
    linestyle='-',
    linewidth=2.0,
    capsize=3,
  )
  ax.set(
    xlabel='Degree k',
    ylabel='Colonization minus random-pair distance',
    title=f'Distance gap by degree at N={fixed_n}',
  )
  sns.despine()
  fig.tight_layout()
  save_figure(fig, path)
  plt.close(fig)


def load_trial_summary(paths):
  trial_df = load_frame(paths['trial_csv'], TRIAL_COLUMNS)
  if trial_df.empty:
    raise RuntimeError(f'No trial data found at {paths["trial_csv"]}')
  summary_df = build_summary(trial_df)
  summary_df.to_csv(paths['summary_csv'], index=False)
  return trial_df, summary_df


def execute_run(config):
  config.validate()
  paths = make_run_paths(config)
  ensure_output_dirs(paths)
  run_simulations(config, paths)
  trial_df, summary_df = load_trial_summary(paths)
  regression_df = pd.DataFrame(columns=REGRESSION_COLUMNS)
  if summary_df['n'].nunique() >= 2:
    regression_df = fit_shared_slope_offset(summary_df, config.degree)
    regression_df.to_csv(paths['regression_csv'], index=False)
  if config.plot_distances:
    regression_to_plot = regression_df if summary_df['n'].nunique() >= 2 else None
    plot_distance_summary(
      summary_df,
      paths['distance_plot'],
      degree=config.degree,
      trials=config.trials,
      regression_df=regression_to_plot,
    )
  if config.plot_shells:
    shell_df = load_frame(paths['shell_csv'], SHELL_COLUMNS)
    if shell_df.empty:
      print(f'Skipping shell plot for degree={config.degree}: no shell data found.')
    else:
      plot_shell_probabilities(shell_df, paths, config.shell_plot_ns)
  return {
    'config': config,
    'paths': paths,
    'trial_df': trial_df,
    'summary_df': summary_df,
    'regression_df': regression_df,
  }


def plot_from_csv(config):
  config.validate()
  paths = make_run_paths(config)
  ensure_output_dirs(paths)
  trial_df, summary_df = load_trial_summary(paths)
  regression_df = pd.DataFrame(columns=REGRESSION_COLUMNS)
  if summary_df['n'].nunique() >= 2:
    regression_df = fit_shared_slope_offset(summary_df, config.degree)
    regression_df.to_csv(paths['regression_csv'], index=False)
  if config.plot_distances:
    plot_distance_summary(
      summary_df,
      paths['distance_plot'],
      degree=config.degree,
      trials=config.trials,
      regression_df=regression_df if not regression_df.empty else None,
    )
  if config.plot_shells:
    shell_df = load_frame(paths['shell_csv'], SHELL_COLUMNS)
    if shell_df.empty:
      raise RuntimeError(f'No shell data found at {paths["shell_csv"]}')
    plot_shell_probabilities(shell_df, paths, config.shell_plot_ns)
  return {
    'config': config,
    'paths': paths,
    'trial_df': trial_df,
    'summary_df': summary_df,
    'regression_df': regression_df,
  }


def run_degree_sweep(degrees, n_min, n_max, num_points, trials, chunk_size, base_seed, max_workers, overwrite):
  degrees = [int(degree) for degree in degrees]
  sweep_rows = []
  for degree in degrees:
    print(f'Running varying-N sweep for degree={degree}')
    run_result = execute_run(RunConfig(
      degree=degree,
      n_min=n_min,
      n_max=n_max,
      num_points=num_points,
      trials=trials,
      chunk_size=chunk_size,
      base_seed=base_seed,
      max_workers=max_workers,
      overwrite=overwrite,
      collect_shells=False,
      plot_shells=False,
      plot_distances=True,
    ))
    regression_df = run_result['regression_df']
    if regression_df.empty:
      raise RuntimeError(f'No regression fit produced for degree={degree}.')
    sweep_rows.append(regression_df.iloc[0].to_dict())
  offset_df = pd.DataFrame(sweep_rows, columns=REGRESSION_COLUMNS).sort_values('degree').reset_index(drop=True)
  sweep_paths = make_degree_sweep_paths(degrees, n_min, n_max, num_points, trials, base_seed)
  ensure_output_dirs(sweep_paths)
  offset_df.to_csv(sweep_paths['offset_csv'], index=False)
  plot_shared_slope_offsets(offset_df, sweep_paths['offset_plot'], n_min=n_min, n_max=n_max)
  print(f'Saved shared-slope offset summary to {sweep_paths["offset_csv"]}')
  return offset_df, sweep_paths


def run_fixed_n_degree_sweep(degrees, n, trials, chunk_size, base_seed, max_workers, overwrite):
  degrees = [int(degree) for degree in degrees]
  sweep_rows = []
  for degree in degrees:
    print(f'Running fixed-N sweep for degree={degree}, N={n}')
    run_result = execute_run(RunConfig(
      degree=degree,
      n_min=n,
      n_max=n,
      num_points=1,
      trials=trials,
      chunk_size=chunk_size,
      base_seed=base_seed,
      max_workers=max_workers,
      overwrite=overwrite,
      collect_shells=False,
      plot_shells=False,
      plot_distances=False,
    ))
    sweep_rows.append(build_fixed_n_degree_row(run_result['trial_df'], degree=degree, n=n))
  fixed_df = pd.DataFrame(sweep_rows, columns=FIXED_N_SWEEP_COLUMNS).sort_values('degree').reset_index(drop=True)
  sweep_paths = make_fixed_n_sweep_paths(n=n, degrees=degrees, trials=trials, base_seed=base_seed)
  ensure_output_dirs(sweep_paths)
  fixed_df.to_csv(sweep_paths['summary_csv'], index=False)
  plot_fixed_n_distances_by_degree(fixed_df, sweep_paths['distance_plot'], fixed_n=n)
  plot_fixed_n_gap_by_degree(fixed_df, sweep_paths['gap_plot'], fixed_n=n)
  print(f'Saved fixed-N degree summary to {sweep_paths["summary_csv"]}')
  return fixed_df, sweep_paths


def run_requested_analysis(args):
  varying_degrees = args.degrees or REQUESTED_DEGREES
  fixed_n = REQUESTED_FIXED_N if args.fixed_n is None else args.fixed_n
  fixed_degrees = args.fixed_degrees or REQUESTED_FIXED_N_DEGREES
  offset_df, offset_paths = run_degree_sweep(
    degrees=varying_degrees,
    n_min=args.n_min,
    n_max=args.n_max,
    num_points=args.num_points,
    trials=args.trials,
    chunk_size=args.chunk_size,
    base_seed=args.base_seed,
    max_workers=args.max_workers,
    overwrite=args.overwrite,
  )
  fixed_df, fixed_paths = run_fixed_n_degree_sweep(
    degrees=fixed_degrees,
    n=fixed_n,
    trials=args.trials,
    chunk_size=args.chunk_size,
    base_seed=args.base_seed,
    max_workers=args.max_workers,
    overwrite=args.overwrite,
  )
  print('Requested analysis complete.')
  print(f'Offset summary CSV: {offset_paths["offset_csv"]}')
  print(f'Offset plot: {offset_paths["offset_plot"]}')
  print(f'Fixed-N summary CSV: {fixed_paths["summary_csv"]}')
  print(f'Fixed-N distance plot: {fixed_paths["distance_plot"]}')
  print(f'Fixed-N gap plot: {fixed_paths["gap_plot"]}')
  return offset_df, fixed_df


def build_parser():
  parser = argparse.ArgumentParser(
    description='Simulate and plot random-regular colonization distances.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument('--requested', action='store_true', help='Run the requested sweeps for this analysis.')
  parser.add_argument('--degrees', nargs='+', type=int, help='Degrees for the varying-N sweep.')
  parser.add_argument('--fixed-n', type=int, help='N for the fixed-N degree sweep.')
  parser.add_argument('--fixed-degrees', nargs='+', type=int, help='Degrees for the fixed-N sweep.')
  parser.add_argument('--degree', type=int, default=D, help='Degree for the legacy single-degree workflow.')
  parser.add_argument('--n-min', type=int, default=N_MIN, help='Minimum N for varying-N runs.')
  parser.add_argument('--n-max', type=int, default=N_MAX, help='Maximum N for varying-N runs.')
  parser.add_argument('--num-points', type=int, default=NUM_POINTS, help='Number of log-spaced N samples.')
  parser.add_argument('--trials', type=int, default=TRIALS, help='Trials per N value.')
  parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Trials per worker chunk.')
  parser.add_argument('--base-seed', type=int, default=BASE_SEED, help='Base RNG seed.')
  parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Process pool size.')
  parser.add_argument('--overwrite', action='store_true', help='Regenerate matching cached outputs.')
  parser.add_argument(
    '--legacy-mode',
    choices=['simulate', 'plot'],
    help='Use the legacy single-degree simulate/plot workflow.',
  )
  parser.add_argument(
    '--shell-plot-ns',
    nargs='+',
    type=int,
    help='Explicit N values to use for legacy shell plotting.',
  )
  parser.add_argument('--skip-shells', action='store_true', help='Skip shell data collection and shell plots.')
  parser.add_argument(
    '--skip-distance-plot',
    action='store_true',
    help='Skip the per-degree distance plot in legacy mode.',
  )
  return parser


def main():
  parser = build_parser()
  args = parser.parse_args()

  if args.requested:
    run_requested_analysis(args)
    return

  if args.degrees:
    run_degree_sweep(
      degrees=args.degrees,
      n_min=args.n_min,
      n_max=args.n_max,
      num_points=args.num_points,
      trials=args.trials,
      chunk_size=args.chunk_size,
      base_seed=args.base_seed,
      max_workers=args.max_workers,
      overwrite=args.overwrite,
    )
    if args.fixed_n is not None:
      fixed_degrees = args.fixed_degrees or REQUESTED_FIXED_N_DEGREES
      run_fixed_n_degree_sweep(
        degrees=fixed_degrees,
        n=args.fixed_n,
        trials=args.trials,
        chunk_size=args.chunk_size,
        base_seed=args.base_seed,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
      )
    return

  if args.fixed_n is not None:
    fixed_degrees = args.fixed_degrees or REQUESTED_FIXED_N_DEGREES
    run_fixed_n_degree_sweep(
      degrees=fixed_degrees,
      n=args.fixed_n,
      trials=args.trials,
      chunk_size=args.chunk_size,
      base_seed=args.base_seed,
      max_workers=args.max_workers,
      overwrite=args.overwrite,
    )
    return

  config = RunConfig(
    degree=args.degree,
    n_min=args.n_min,
    n_max=args.n_max,
    num_points=args.num_points,
    trials=args.trials,
    chunk_size=args.chunk_size,
    base_seed=args.base_seed,
    max_workers=args.max_workers,
    overwrite=args.overwrite,
    shell_plot_ns=args.shell_plot_ns if args.shell_plot_ns is not None else SHELL_PLOT_NS,
    collect_shells=not args.skip_shells,
    plot_shells=not args.skip_shells,
    plot_distances=not args.skip_distance_plot,
  )
  legacy_mode = args.legacy_mode or MODE
  if legacy_mode == 'simulate':
    execute_run(config)
    return
  if legacy_mode == 'plot':
    plot_from_csv(config)
    return
  raise ValueError(f'Unsupported legacy mode={legacy_mode!r}. Use simulate or plot.')


if __name__ == '__main__':
  main()
