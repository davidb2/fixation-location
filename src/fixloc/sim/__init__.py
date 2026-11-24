'''The C++ simulator sweepsim and its files. Build it with: make -C src/fixloc/sim'''
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, 'sweepsim')


def build():
  '''Compile sweepsim if the binary is missing and return its path.'''
  if not os.path.exists(BINARY):
    subprocess.run(['make', '-C', HERE], check=True, capture_output=True)
  return BINARY


def edges(G):
  '''Number of vertices and directed edges, both directions for undirected graphs, vertices relabeled 0..N-1.'''
  nodes = sorted(G.nodes())
  idx = {u: i for i, u in enumerate(nodes)}
  E = []
  for a, b in G.edges():
    E.append((idx[a], idx[b]))
    if not G.is_directed():
      E.append((idx[b], idx[a]))
  return len(nodes), E


def write_edges(N, E, path):
  '''Write N, then one directed edge a b per line.'''
  with open(path, 'w') as f:
    f.write(f'{N}\n')
    for a, b in E:
      f.write(f'{a} {b}\n')


def read_meta(path):
  '''N, checkpoint times, and per initial vertex the numbers of runs, fixations, and capped runs.'''
  lines = open(path + '.meta').read().split('\n')
  head = lines[0].split()
  N = int(head[1])
  checks = [int(x) for x in head[head.index('checkpoints') + 1 :]]
  rows = np.array([[int(x) for x in line.split()] for line in lines[2:] if line.strip()])
  return N, checks, rows[:, 1], rows[:, 2], rows[:, 3]


def read_records(path):
  '''The runs ending in fixation: initial vertex u, fixation location v, fixation time t, and the states.'''
  N, checks, runs, fixes, capped = read_meta(path)
  C = len(checks)
  nb = (N + 7) // 8
  dt = np.dtype([('u', '<i4'), ('v', '<i4'), ('t', '<i8')] + ([('snap', 'u1', (C, nb))] if C else []))
  rec = np.fromfile(path, dtype=dt)
  out = dict(N=N, checks=checks, runs=runs, fixes=fixes, capped=capped, u=rec['u'], v=rec['v'], t=rec['t'])
  if C:
    out['state'] = np.unpackbits(rec['snap'], axis=2, bitorder='little')[:, :, :N]
  return out


def run(G, r, K, seed, times=(), max_runs=None):
  '''Simulate until K runs from every initial vertex end in fixation, and return read_records of the output.'''
  binary = build()
  with tempfile.TemporaryDirectory() as d:
    N, E = edges(G)
    ef, of = os.path.join(d, 'edges.txt'), os.path.join(d, 'out.bin')
    write_edges(N, E, ef)
    args = [binary, ef, str(r), str(K), str(seed), of]
    if max_runs is not None or times:
      args.append(str(max_runs if max_runs is not None else K * 100000))
    args += [str(t) for t in times]
    subprocess.run(args, check=True)
    return read_records(of)
