'''Figure S3: guessing the initial vertex on graphs with N = 10. Run: python -m fixloc.make.figS3'''
from multiprocessing import Pool

from fixloc.make import dump
from fixloc.origin import families, scores

N = 10
RS = (1.0, 1.5, 2.0, 5.0)
COLS = ('prior_best', 'acc_duration', 'acc_farthest', 'acc_location')


def compute():
  '''One row of scores for every graph of families(N) and every r in RS.'''
  tasks = [(f, s, G, r, True) for f, s, G in families(N) for r in RS]
  with Pool(8) as p:
    rows = [x for x in p.imap_unordered(scores, tasks) if x is not None]
  rows.sort(key=lambda x: (x['family'], x['seed'], x['r']))
  return rows


def write(rows):
  keep = [dict(family=x['family'], seed=x['seed'], **{k: x[k] for k in COLS}) for x in rows if x['r'] == 2.0]
  dump('figS3', 'a', dict(r=2.0, rows=keep))


if __name__ == '__main__':
  write(compute())
