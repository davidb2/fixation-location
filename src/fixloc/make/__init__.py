'''One module per figure. Each module computes the data of its panels and writes data.json into each panel folder.'''
import json
import os

FIGS = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'figs'))


def dump(fig, panel, data):
  path = os.path.join(FIGS, fig, panel, 'data.json')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump(data, f)
  return path
