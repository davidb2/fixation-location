'''SI section "Directed cycles". Run: python -m fixloc.make.si.directed_cycles (N = 160 takes about a minute)'''
from fixloc import walk
from fixloc.make.si import dump

SECTION = 'directed_cycles'


def neutral(Ns=(10, 20, 40, 80, 160)):
  return {str(N): walk.neutral_deviation(N) for N in Ns}


if __name__ == '__main__':
  dump(SECTION, 'directed_neutral_check.json', neutral())
