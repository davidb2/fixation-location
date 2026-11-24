'''One module per section of the SI. Each module computes the data behind the numbers quoted in its section and writes
the data into src/figs/si_numbers/<section>/. The notebook of the section prints every quoted number.

Per-graph tables too large for the repository go to the folder raw/ of the section, which git ignores.
'''
import json
import os
import re

from fixloc.make import FIGS

NUMBERS = os.path.join(FIGS, 'si_numbers')


def path(section, name):
  p = os.path.join(NUMBERS, section, name)
  os.makedirs(os.path.dirname(p), exist_ok=True)
  return p


def dump(section, name, data):
  with open(path(section, name), 'w') as f:
    json.dump(data, f, indent=1)
  return path(section, name)


def load(section, name):
  return json.load(open(path(section, name)))


WORDS = [
  ('{,}', ','), ('\\%', '%'), ('\\times', ' x '), ('\\leq', '<='), ('\\geq', '>='), ('\\ge', '>='), ('\\le', '<='),
  ('\\approx', ' about '), ('\\setminus', ' minus '), ('\\{', '{'), ('\\}', '}'), ('\\ldots', '...'),
  ('\\coloneqq', ':='), ('--', '-'), ("\\'", ''), ('\\H', ''), ('\\,', ' '), ('\\ell', 'l'), ('\\sum', 'sum'),
  ('\\max', 'max'), ('\\pi', 'pi'), ('\\rho', 'rho'), ('\\sqrt', 'sqrt'), ('\\theta', 'theta'), ('\\mathrm', ''),
  ('\\to', ' -> '),
]


def plain(latex):
  '''An SI sentence without the LaTeX markup.'''
  s = re.sub(r'\\(Cref|cite|ref)\{[^}]*\}', '[ref]', latex)
  for a, b in WORDS:
    s = s.replace(a, b)
  s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
  s = re.sub(r'\\[a-zA-Z]+', '', s)
  return re.sub(r'\s+', ' ', s.replace('$', '').replace('{', '').replace('}', '').replace('~', ' ')).strip()


def quote(sentence, *numbers):
  '''Print an SI sentence and, for each quoted number (si, value, fmt), the value formatted at the precision of the SI.

  fmt is a format string or a function. The last column states whether the formatted value equals the quoted value.'''
  print(plain(sentence))
  for si, value, fmt in numbers:
    got = fmt(value) if callable(fmt) else fmt.format(value)
    raw = f'{value:.6g}' if isinstance(value, float) else str(value)
    print(f'    SI {si:>10}   computed {got:>10}   ({raw})   {"ok" if got == si else "DIFFERS"}')
  print()
