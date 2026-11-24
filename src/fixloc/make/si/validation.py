'''SI section "Validation". Run: python -m fixloc.make.si.validation

Parts, run by name:
  solver      the exact solver against the Moran formula and the isothermal theorem (seconds)
  simulator   the simulator against the exact solver, 20,000 fixation trajectories per initial vertex (minutes)
  python      the Python simulator of fixloc.exact against the exact solver, 40,000 runs per graph (minutes)
'''
import sys

from fixloc import validate
from fixloc.make.si import dump

SECTION = 'validation'

if __name__ == '__main__':
  parts = sys.argv[1:] or ['solver']
  if 'solver' in parts:
    dump(SECTION, 'check_solver.json', validate.solver())
  if 'simulator' in parts:
    dump(SECTION, 'check_sim.json', validate.simulator())
  if 'python' in parts:
    dump(SECTION, 'check_python_sim.json', validate.monte_carlo())
