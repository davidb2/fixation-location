'''Figure S1: the Charles River basin as a population structure. Run: python -m fixloc.make.figS1 (under a second)

Panel a is the temperature of each vertex. Panel b is the fixation location probability at r = 2 for a uniformly random
initial vertex, conditional on fixation, from 4,000 simulated fixation events per initial vertex. Both panels hold the
polyline and the stream order of every reach, which set the drawing, and the mouth of the river, which is circled.

The inputs are committed in src/figs/si_numbers/charles/, next to the notebook of the SI numbers:
  charles_basin.json              reaches and polylines, python -m fixloc.make.si.charles_basin extract DIR
  charles_network.json            the population structure, python -m fixloc.make.si.charles_basin
  charles_net.txt                 its edge list for sweepsim, python -m fixloc.make.si.charles_basin
  sim/charles_watch_rank.json     network 60 at r = 2, 1.5, and 5, from python -m fixloc.large watch with ARRAYS=1
  sim/charles_watch_rank_r1.json  network 61 at r = 1, the same
  sim/summary_r<r>.json           python -m fixloc.large analyze, run by run_network.sh and run_network_capped.sh
  sim/main_r1.0.bin.meta          runs and fixation events per initial vertex of the neutral run, from sweepsim
  sim/ck_r1.0.bin.meta            the same for the neutral run that records the mutant sets at three times
The per-vertex arrays of a watch file are the fixation location probability last, the probability pi of each initial
vertex given fixation, and the simulated fixation probability fix_prob.
src/fixloc/make/si/charles_basin.py describes the download, the extraction, and the cluster runs.
'''
from fixloc.make import dump
from fixloc.make.si import load

SECTION = 'charles'
R = 2.0


def geometry():
  '''The reaches in vertex order: HydroRIVERS ids, stream orders, polylines, and the mouth of the river.'''
  basin = load(SECTION, 'charles_basin.json')
  net = load(SECTION, 'charles_network.json')
  lines = [basin['lines'][str(rid)] for rid in net['ids']]
  return dict(N=net['N'], ids=net['ids'], order=net['order'], mouth=basin['mouth'], lines=lines)


def compute():
  net = load(SECTION, 'charles_network.json')
  w = load(SECTION, 'sim/charles_watch_rank.json')[f'60:{R}']
  assert w['N'] == net['N']
  g = geometry()
  return dict(
    a=dict(g, theta=net['theta']),
    b=dict(g, r=R, K=w['takeovers'] // w['N'], P_last=w['last']),
  )


def write(res):
  dump('figS1', 'a', res['a'])
  dump('figS1', 'b', res['b'])


if __name__ == '__main__':
  write(compute())
