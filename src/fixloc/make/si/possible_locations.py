'''SI section "Possible locations of fixation". Run: python -m fixloc.make.si.possible_locations

Parts, run by name:
  directed     the colonization proposition against exhaustive search on directed graphs (seconds)
  undirected   reachability on the 986 connected undirected graphs with N = 5 to 7 (about a minute)
The exact fixation location probabilities of the graph with twelve vertices are the data of Figure S2a.
'''
import sys

from fixloc import possible
from fixloc.make.si import dump

SECTION = 'possible_locations'

if __name__ == '__main__':
  parts = sys.argv[1:] or ['directed']
  if 'directed' in parts:
    dump(SECTION, 'colonization_directed.json', possible.check_directed())
  if 'undirected' in parts:
    dump(SECTION, 'impossible_reachability.json', possible.check_undirected())
