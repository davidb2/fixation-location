'''Figure S2: impossible fixation locations on twelve vertices with directed edges. Run: python -m fixloc.make.figS2'''
from fixloc import monitor
from fixloc.make import dump


def compute():
  return monitor.example12()


def write(res):
  dump('figS2', 'a', {k: res[k] for k in ('nodes', 'edges', 'initial', 'r', 'P_last')})


if __name__ == '__main__':
  write(compute())
