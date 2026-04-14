import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIDE_LENGTHS = list(range(3, 22, 2))
TRIALS = 10_000
CHUNK_SIZE = 250
BASE_SEED = 123
MAX_WORKERS = min(12, os.cpu_count() or 12)
OVERWRITE = False

TRIAL_COLUMNS = [
  'n',
  'N',
  'trial_id',
  'start_vertex',
  'finish_vertex',
  'start_finish_distance',
  'random_pair_distance',
]

SUMMARY_COLUMNS = [
  'n',
  'N',
  'metric',
  'metric_label',
  'count',
  'mean_distance',
  'std_distance',
  'se_distance',
]

REGRESSION_COLUMNS = [
  'x_axis',
  'metric',
  'metric_label',
  'intercept',
  'intercept_se',
  'slope',
  'slope_se',
  'r_squared',
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

X_AXIS_LABELS = {
  'n': 'Side length n',
  'N': 'Total vertices N = n^2',
}


@dataclass(frozen=True)
class RunConfig:
  side_lengths: Sequence[int]
  trials: int = TRIALS
  chunk_size: int = CHUNK_SIZE
  base_seed: int = BASE_SEED
  max_workers: int = MAX_WORKERS
  overwrite: bool = OVERWRITE

  def normalized_side_lengths(self):
    return tuple(sorted(int(n) for n in self.side_lengths))

  def validate(self):
    side_lengths = self.normalized_side_lengths()
    if not side_lengths:
      raise ValueError('At least one odd side length is required.')
    if any(n <= 0 for n in side_lengths):
      raise ValueError(f'Side lengths must be positive; received {side_lengths}.')
    if any(n % 2 == 0 for n in side_lengths):
      raise ValueError(f'Periodic-grid side lengths must be odd; received {side_lengths}.')
    if self.trials <= 0:
      raise ValueError(f'trials must be positive; received {self.trials}.')
    if self.chunk_size <= 0:
      raise ValueError(f'chunk_size must be positive; received {self.chunk_size}.')
    if self.max_workers <= 0:
      raise ValueError(f'max_workers must be positive; received {self.max_workers}.')


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


def make_run_tag(config):
  return (
    f'n{summarize_integer_sequence(config.normalized_side_lengths())}'
    f'_trials{config.trials}_seed{config.base_seed}'
  )


def make_paths(config):
  tag = make_run_tag(config)
  data_dir = Path('data/figure_5_periodic_grid')
  figure_dir = Path('figs/figure_5_periodic_grid')
  return {
    'tag': tag,
    'data_dir': data_dir,
    'figure_dir': figure_dir,
    'trial_csv': data_dir / f'{tag}_trials.csv',
    'summary_csv': data_dir / f'{tag}_summary.csv',
    'regression_csv': data_dir / f'{tag}_linear_fits.csv',
    'distance_plot_n': figure_dir / f'{tag}_distances_vs_n.png',
    'distance_plot_N': figure_dir / f'{tag}_distances_vs_total_vertices.png',
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


def center_vertex(n):
  return int(n) * (int(n) // 2) + (int(n) // 2)


def build_square_torus_neighbors(n):
  n = int(n)
  neighbors = []
  for row in range(n):
    for col in range(n):
      up = ((row - 1) % n) * n + col
      down = ((row + 1) % n) * n + col
      left = row * n + ((col - 1) % n)
      right = row * n + ((col + 1) % n)
      neighbors.append((up, down, left, right))
  return tuple(neighbors)


def torus_manhattan_distance(n, u, v):
  n = int(n)
  row_u, col_u = divmod(int(u), n)
  row_v, col_v = divmod(int(v), n)
  row_delta = abs(row_u - row_v)
  col_delta = abs(col_u - col_v)
  return int(min(row_delta, n - row_delta) + min(col_delta, n - col_delta))


def run_colonization_trial(neighbors, start, rng):
  # main colonization logic
  num_nodes = len(neighbors)
  occupied = [int(start)]
  occupied_mask = bytearray(num_nodes)
  occupied_mask[int(start)] = 1
  finish = int(start)
  while len(occupied) < num_nodes:
    parent = occupied[rng.randrange(len(occupied))]
    child = neighbors[parent][rng.randrange(4)]
    if not occupied_mask[child]:
      occupied_mask[child] = 1
      occupied.append(child)
      finish = child
  return finish


def load_frame(path, columns):
  if not path.exists():
    return pd.DataFrame(columns=columns)
  frame = pd.read_csv(path)
  for column in columns:
    if column not in frame.columns:
      frame[column] = pd.NA
  return frame[columns]


def build_summary(trial_df):
  if trial_df.empty:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)
  long_df = trial_df.melt(
    id_vars=['n', 'N', 'trial_id'],
    value_vars=['start_finish_distance', 'random_pair_distance'],
    var_name='metric',
    value_name='distance',
  )
  summary_df = long_df.groupby(['n', 'N', 'metric'], as_index=False).agg(
    count=('distance', 'size'),
    mean_distance=('distance', 'mean'),
    std_distance=('distance', 'std'),
  )
  summary_df['std_distance'] = summary_df['std_distance'].fillna(0.0)
  summary_df['se_distance'] = summary_df['std_distance'] / np.sqrt(summary_df['count'])
  summary_df['metric_label'] = summary_df['metric'].map(
    {metric: style['label'] for metric, style in METRIC_STYLES.items()}
  )
  return summary_df[SUMMARY_COLUMNS].sort_values(['metric', 'n']).reset_index(drop=True)


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


def simulate_chunk(args):
  n, chunk_trials, seed, trial_offset = args
  rng = random.Random(int(seed))
  n = int(n)
  num_nodes = n * n
  start = center_vertex(n)
  neighbors = build_square_torus_neighbors(n)
  trial_rows = []
  for local_trial_id in range(int(chunk_trials)):
    trial_id = int(trial_offset) + local_trial_id
    finish = run_colonization_trial(neighbors, start, rng)
    u = rng.randrange(num_nodes)
    v = rng.randrange(num_nodes - 1)
    if v >= u:
      v += 1
    trial_rows.append({
      'n': n,
      'N': num_nodes,
      'trial_id': int(trial_id),
      'start_vertex': int(start),
      'finish_vertex': int(finish),
      'start_finish_distance': torus_manhattan_distance(n, start, finish),
      'random_pair_distance': torus_manhattan_distance(n, u, v),
    })
  return trial_rows


def simulate_for_n(config, n):
  chunk_args = []
  remaining = int(config.trials)
  chunk_index = 0
  trial_offset = 0
  while remaining > 0:
    chunk_trials = min(int(config.chunk_size), remaining)
    seed = int(config.base_seed) + int(n) * 1_000_000 + chunk_index
    chunk_args.append((int(n), int(chunk_trials), int(seed), int(trial_offset)))
    remaining -= chunk_trials
    trial_offset += chunk_trials
    chunk_index += 1
  trial_rows = []
  for chunk_trial_rows in map_simulation_chunks(chunk_args, config.max_workers):
    trial_rows.extend(chunk_trial_rows)
  return pd.DataFrame.from_records(trial_rows, columns=TRIAL_COLUMNS)


def fit_metric_line(metric_df, x_axis):
  if metric_df.empty:
    raise RuntimeError(f'No metric data available for x_axis={x_axis}.')
  if metric_df[x_axis].nunique() < 2:
    raise RuntimeError('A best-fit line needs at least two distinct x values.')
  metric_df = metric_df.sort_values(x_axis).reset_index(drop=True)
  x = metric_df[x_axis].astype(float).to_numpy()
  y = metric_df['mean_distance'].astype(float).to_numpy()
  se = metric_df['se_distance'].fillna(0.0).astype(float).to_numpy()
  weights = np.ones_like(y)
  positive_se = se > 0
  weights[positive_se] = 1.0 / np.square(se[positive_se])
  design = np.column_stack([np.ones_like(x), x])
  weighted_design = design * np.sqrt(weights)[:, None]
  weighted_y = y * np.sqrt(weights)
  coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
  intercept, slope = coefficients
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
  return {
    'x_axis': x_axis,
    'metric': metric_df.iloc[0]['metric'],
    'metric_label': metric_df.iloc[0]['metric_label'],
    'intercept': float(intercept),
    'intercept_se': float(np.sqrt(max(covariance[0, 0], 0.0))),
    'slope': float(slope),
    'slope_se': float(np.sqrt(max(covariance[1, 1], 0.0))),
    'r_squared': float(r_squared),
  }


def fit_linear_trends(summary_df, x_axis):
  if summary_df.empty:
    raise RuntimeError(f'No summary data available for x_axis={x_axis}.')
  rows = []
  for metric in METRIC_STYLES:
    metric_df = summary_df[summary_df['metric'] == metric].copy()
    rows.append(fit_metric_line(metric_df, x_axis))
  return pd.DataFrame(rows, columns=REGRESSION_COLUMNS)


def write_outputs(trial_df, summary_df, regression_df, paths):
  trial_df.sort_values(['n', 'trial_id']).to_csv(paths['trial_csv'], index=False)
  summary_df.to_csv(paths['summary_csv'], index=False)
  regression_df.to_csv(paths['regression_csv'], index=False)


def run_simulations(config, paths):
  side_lengths = config.normalized_side_lengths()
  if config.overwrite:
    trial_df = pd.DataFrame(columns=TRIAL_COLUMNS)
  else:
    trial_df = load_frame(paths['trial_csv'], TRIAL_COLUMNS)
  print(f'Using side lengths n={list(side_lengths)}')
  print(f'Workers={config.max_workers}, trials per n={config.trials}')
  print(f'Writing trial data to {paths["trial_csv"]}')
  for n in side_lengths:
    existing_trial_count = 0 if trial_df.empty else int((trial_df['n'] == n).sum())
    if not config.overwrite and existing_trial_count == config.trials:
      print(f'Skipping n={n}: found {existing_trial_count} saved trials')
      continue
    if existing_trial_count > 0:
      trial_df = trial_df[trial_df['n'] != n].copy()
    new_trial_df = simulate_for_n(config, n)
    trial_df = pd.concat([trial_df, new_trial_df], ignore_index=True)
    summary_df = build_summary(trial_df)
    regression_rows = []
    if summary_df['n'].nunique() >= 2:
      regression_rows.extend(fit_linear_trends(summary_df, 'n').to_dict('records'))
      regression_rows.extend(fit_linear_trends(summary_df, 'N').to_dict('records'))
    regression_df = pd.DataFrame(regression_rows, columns=REGRESSION_COLUMNS)
    write_outputs(trial_df, summary_df, regression_df, paths)
    current_summary = summary_df[summary_df['n'] == n].set_index('metric')
    process_mean = current_summary.at['start_finish_distance', 'mean_distance']
    process_se = current_summary.at['start_finish_distance', 'se_distance']
    baseline_mean = current_summary.at['random_pair_distance', 'mean_distance']
    baseline_se = current_summary.at['random_pair_distance', 'se_distance']
    print(
      f'n={n:2d}, N={n * n:3d} | '
      f'colonization mean={process_mean:.4f} (SE={process_se:.4f}) | '
      f'random mean={baseline_mean:.4f} (SE={baseline_se:.4f})'
    )
  print(f'Finished simulation data in {paths["data_dir"]}')


def plot_distance_summary(summary_df, regression_df, path, x_axis, trials):
  if summary_df.empty:
    raise RuntimeError('No summary data available to plot.')
  sns.set_theme(style='ticks', context='talk')
  fig, ax = plt.subplots(figsize=(8.5, 5.5))
  fit_row = None
  for metric, style in METRIC_STYLES.items():
    metric_df = summary_df[summary_df['metric'] == metric].sort_values(x_axis)
    ax.plot(
      metric_df[x_axis],
      metric_df['mean_distance'],
      marker=style['marker'],
      linestyle=style['linestyle'],
      color=style['color'],
      linewidth=2.2,
      label=style['label'],
    )
    ax.errorbar(
      metric_df[x_axis],
      metric_df['mean_distance'],
      yerr=metric_df['se_distance'],
      fmt='none',
      color=style['color'],
      capsize=3,
      linewidth=1.2,
    )
    fit_row = None
    if not regression_df.empty:
      fit_rows = regression_df[
        (regression_df['x_axis'] == x_axis) & (regression_df['metric'] == metric)
      ]
      if not fit_rows.empty:
        fit_row = fit_rows.iloc[0]
    if fit_row is not None and metric_df[x_axis].nunique() >= 2:
      x_grid = np.linspace(metric_df[x_axis].min(), metric_df[x_axis].max(), 200)
      ax.plot(
        x_grid,
        fit_row['intercept'] + fit_row['slope'] * x_grid,
        color=style['color'],
        linestyle=':',
        linewidth=2.0,
        label=f'{style["label"]} (best-fit line)',
      )
  ax.set(
    xlabel=X_AXIS_LABELS[x_axis],
    ylabel='Average torus distance',
    title=f'Periodic n x n torus distances vs {x_axis} (center start, trials={trials})',
  )
  ax.legend(frameon=False)
  sns.despine()
  fig.tight_layout()
  save_figure(fig, path)
  plt.close(fig)


def load_results(paths):
  trial_df = load_frame(paths['trial_csv'], TRIAL_COLUMNS)
  if trial_df.empty:
    raise RuntimeError(f'No trial data found at {paths["trial_csv"]}')
  summary_df = build_summary(trial_df)
  regression_rows = []
  if summary_df['n'].nunique() >= 2:
    regression_rows.extend(fit_linear_trends(summary_df, 'n').to_dict('records'))
    regression_rows.extend(fit_linear_trends(summary_df, 'N').to_dict('records'))
  regression_df = pd.DataFrame(regression_rows, columns=REGRESSION_COLUMNS)
  write_outputs(trial_df, summary_df, regression_df, paths)
  return trial_df, summary_df, regression_df


def execute_run(config):
  config.validate()
  paths = make_paths(config)
  ensure_output_dirs(paths)
  run_simulations(config, paths)
  _, summary_df, regression_df = load_results(paths)
  plot_distance_summary(summary_df, regression_df, paths['distance_plot_n'], x_axis='n', trials=config.trials)
  plot_distance_summary(summary_df, regression_df, paths['distance_plot_N'], x_axis='N', trials=config.trials)
  return paths


def plot_from_csv(config):
  config.validate()
  paths = make_paths(config)
  ensure_output_dirs(paths)
  _, summary_df, regression_df = load_results(paths)
  plot_distance_summary(summary_df, regression_df, paths['distance_plot_n'], x_axis='n', trials=config.trials)
  plot_distance_summary(summary_df, regression_df, paths['distance_plot_N'], x_axis='N', trials=config.trials)
  return paths


def print_slope_summary(regression_df):
  if regression_df.empty:
    print('No fitted slopes were produced yet.')
    return
  for x_axis in ['n', 'N']:
    axis_df = regression_df[regression_df['x_axis'] == x_axis].sort_values('metric')
    if axis_df.empty:
      continue
    print(f'Best-fit slopes vs {x_axis}:')
    for _, row in axis_df.iterrows():
      print(
        f'  {row["metric_label"]}: slope={row["slope"]:.8f} '
        f'(SE={row["slope_se"]:.8f}, R^2={row["r_squared"]:.6f})'
      )


def build_parser():
  parser = argparse.ArgumentParser(
    description='Simulate periodic square-grid colonization distances and plot them.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    '--side-lengths',
    nargs='+',
    type=int,
    default=SIDE_LENGTHS,
    help='Odd periodic-grid side lengths n to simulate.',
  )
  parser.add_argument('--trials', type=int, default=TRIALS, help='Trials per side length.')
  parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Trials per worker chunk.')
  parser.add_argument('--base-seed', type=int, default=BASE_SEED, help='Base RNG seed.')
  parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Process pool size.')
  parser.add_argument('--overwrite', action='store_true', help='Regenerate matching cached outputs.')
  parser.add_argument('--plot-only', action='store_true', help='Skip simulation and replot from saved CSVs.')
  return parser


def main():
  parser = build_parser()
  args = parser.parse_args()
  config = RunConfig(
    side_lengths=args.side_lengths,
    trials=args.trials,
    chunk_size=args.chunk_size,
    base_seed=args.base_seed,
    max_workers=args.max_workers,
    overwrite=args.overwrite,
  )
  if args.plot_only:
    paths = plot_from_csv(config)
  else:
    paths = execute_run(config)
  regression_df = load_frame(paths['regression_csv'], REGRESSION_COLUMNS)
  print_slope_summary(regression_df)
  print(f'Trial CSV: {paths["trial_csv"]}')
  print(f'Summary CSV: {paths["summary_csv"]}')
  print(f'Regression CSV: {paths["regression_csv"]}')
  print(f'Figure vs n: {paths["distance_plot_n"]}')
  print(f'Figure vs N: {paths["distance_plot_N"]}')


if __name__ == '__main__':
  main()
