'''SI section "Fixation location, fixation probability, and fixation time". Run: python -m fixloc.make.si.factorization

The exact table of every connected undirected graph with N = 5 to 9 vertices goes to raw/allgraphs_N{N}_r{r}.npz.
With N = 9 the table takes about an hour on 50 cores, so it runs on a cluster with python -m fixloc.allgraphs chunk and
gather.
Parts, run by name:
  tables N [N ...]    exact tables for the given N at r = 1.25, 2, and 10 (FIXLOC_R=1 for neutral evolution)
  summary             allgraphs_summary.json from the tables in raw/ and the cluster summaries for N = 9
  directed            every strongly connected directed graph with N = 5, from nauty geng and directg (minutes)
'''
import glob
import json
import subprocess
import sys

from fixloc import allgraphs
from fixloc.make.si import dump, path

SECTION = 'factorization'


def tables(Ns, rs=allgraphs.R_VALUES):
  for N in Ns:
    g6s = allgraphs.geng(N)
    for r in rs:
      allgraphs.save(path(SECTION, f'raw/allgraphs_N{N}_r{r:g}.npz'), g6s, allgraphs.compute(g6s, r))


def summary():
  '''Summaries of the tables in raw/, completed by the cluster summaries raw/allgraphs_N*_summary.json.'''
  out = allgraphs.summarize_dir(path(SECTION, 'raw'))
  for f in sorted(glob.glob(path(SECTION, 'raw/allgraphs_N*_summary.json'))):
    have = {(e['N'], e['r']) for e in out}
    out += [e for e in json.load(open(f)) if (e['N'], e['r']) not in have]
  return sorted(out, key=lambda e: (e['r'] != 1, e['N'], e['r']))


def directed(N=5):
  src = path(SECTION, f'raw/digraphs_N{N}.txt')
  g = subprocess.run(['geng', '-cq', str(N)], capture_output=True).stdout
  open(src, 'wb').write(subprocess.run(['directg', '-T', '-q'], input=g, capture_output=True).stdout)
  return allgraphs.directed(src, N, path(SECTION, 'raw'))


if __name__ == '__main__':
  parts = sys.argv[1:] or ['summary']
  if parts[0] == 'tables':
    tables([int(x) for x in parts[1:]])
  if 'summary' in parts:
    dump(SECTION, 'allgraphs_summary.json', summary())
  if 'directed' in parts:
    dump(SECTION, 'directed_N5_summary.json', directed())
