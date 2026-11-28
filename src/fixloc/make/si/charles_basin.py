'''SI paragraph on the Charles River basin and Figure S1: the river network of the Charles River basin in Massachusetts.

Run from src/: python -m fixloc.make.si.charles_basin [extract SHAPEFILE_DIR]

Without arguments the module writes the population structure from the committed charles_basin.json (under a second).
With extract it first writes charles_basin.json from the HydroRIVERS shapefile (it reads the 100 MB attribute table).
The output goes to src/figs/si_numbers/charles/:
  charles_basin.json     the 77 reaches of the basin with their HydroRIVERS attributes and polylines
  charles_network.json   the population structure: N, q_up, ids, edges, theta, order, upland_skm, outlet
  charles_net.txt        the edge list of the population structure for sweepsim

HydroRIVERS (Lehner and Grill 2013, Hydrological Processes 27:2171) records every river as a list of reaches, the
stretches of river between two confluences, with the downstream reach of each (NEXT_DOWN, 0 at an outlet). The basin is
every reach upstream of the outlet reach nearest to the mouth of the Charles River at Boston Harbor.

Download HydroRIVERS v1.0, North and Central America, as a shapefile: HydroRIVERS_v10_na_shp.zip (66 MB) from
https://www.hydrosheds.org/products/hydrorivers, which links to
https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip. Unzip it outside the repository and pass the
folder that holds HydroRIVERS_v10_na.shp. The module reads only the .shp, .shx, and .dbf files, with the readers below
and numpy. HydroRIVERS is free for scientific, educational, and commercial use under the HydroSHEDS license. Cite
  B. Lehner and G. Grill, Global river hydrography and network routing: baseline data and new approaches to study the
  world's large river systems, Hydrological Processes 27(15):2171-2186 (2013), doi:10.1002/hyp.9740.

The population structure has one vertex per reach, numbered in the breadth-first order from the outlet reach, vertex 0,
which is the root of the tree. An offspring is placed at a neighboring reach with weight DOWN = 2 downstream, toward the
root, and with weight UP = 1 upstream. The edge list repeats every downstream edge twice, because sweepsim places an
offspring at a uniformly chosen out-neighbor.

The simulations in src/figs/si_numbers/charles/sim/ ran on a cluster with sweepsim, run_network.sh, and
run_network_capped.sh as networks 60 and 61 of fixloc.large. From a folder BASE that holds src/:
  make -C src/fixloc/sim
  mkdir -p large/net60 large/net61
  cp src/figs/si_numbers/charles/charles_net.txt large/net60/net.txt
  cp src/figs/si_numbers/charles/charles_net.txt large/net61/net.txt
  echo 2.0 1.5 5.0 > large/net60/rlist.txt
  echo 1.0 > large/net61/rlist.txt
  sbatch --array=60 src/fixloc/make/si/run_network.sh
  sbatch --array=61 src/fixloc/make/si/run_network_capped.sh
  sbatch src/fixloc/make/si/run_charles_T.sh
  cp charles_watch_T.json src/figs/si_numbers/charles/sim/
The runners write summary_r<r>.json into each network folder, and sweepsim writes the .meta files next to the .bin
files. run_charles_T.sh records the mutant sets at the expected fixation time T, conditional on fixation, and writes
charles_watch_T.json, the best vertex to sample at T with standard errors. sim/ holds charles_watch_T.json, the four
summaries, and the .meta files of the main runs (main_r<r>.bin.meta) and of the runs at T (ckT_r<r>.bin.meta).
The main runs have 4,000 fixation events per initial vertex, after a pilot run of 100, and the runs at T have 4,000.
Network 61 runs r = 1 with at most 10^6 runs per initial vertex in each phase. The seed is 100 ID + 10 i + 2 for the
main run and 100 ID + 10 i + 4 for the run at T, with i the position of r in rlist.txt (2.0, 1.5, 5.0 for network 60).
The .meta files record the seeds. run_network.sh also records mutant sets at three quantile times of the fixation time,
which the SI does not use.
'''
import json
import os
import struct
import sys

import numpy as np

from fixloc.make.si import load, path
from fixloc.sim import write_edges

SECTION = 'charles'
MOUTH = (-71.053, 42.372)  # mouth of the Charles River at Boston Harbor, longitude and latitude
FIELDS = {'HYRIV_ID', 'NEXT_DOWN', 'MAIN_RIV', 'LENGTH_KM', 'UPLAND_SKM', 'ORD_STRA', 'DIS_AV_CMS', 'CATCH_SKM'}
SOURCE = 'HydroRIVERS v1.0, North and Central America, Lehner and Grill 2013'
DOWN, UP = 2, 1


def read_dbf(fname, wanted):
  '''The columns named in wanted of a dBASE table, numeric columns as float arrays, and the names of all fields.'''
  with open(fname, 'rb') as fh:
    nrec, hlen, rlen = struct.unpack('<xxxxIHH20x', fh.read(32))
    fields = []
    while True:
      fd = fh.read(32)
      if fd[0] == 0x0D:
        break
      fields.append((fd[:11].split(b'\x00')[0].decode(), chr(fd[11]), fd[16]))
    fh.seek(hlen)
    raw = np.frombuffer(fh.read(nrec * rlen), dtype=np.uint8).reshape(nrec, rlen)
  cols, pos = {}, 1
  for name, ftype, flen in fields:
    if name in wanted:
      txt = raw[:, pos : pos + flen].tobytes().decode('ascii', 'replace')
      vals = [txt[i * flen : (i + 1) * flen].strip() for i in range(nrec)]
      if ftype in 'NF':
        cols[name] = np.array([float(v) if v not in ('', '-') else np.nan for v in vals])
      else:
        cols[name] = np.array(vals)
    pos += flen
  return cols, [f[0] for f in fields]


def read_shx(fname):
  '''Byte offsets and lengths of the records of a shapefile, from its index file.'''
  data = np.fromfile(fname, dtype='>i4', offset=100).reshape(-1, 2)
  return data[:, 0] * 2, data[:, 1] * 2


def read_polyline(fh, offset):
  '''The points of the polyline record at a byte offset of an open shapefile, all parts joined.'''
  fh.seek(offset + 8)
  stype = struct.unpack('<i', fh.read(4))[0]
  assert stype in (3, 13, 23), stype
  fh.read(32)
  nparts, npts = struct.unpack('<ii', fh.read(8))
  fh.read(4 * nparts)
  return np.frombuffer(fh.read(16 * npts), dtype='<f8').reshape(npts, 2)


def extract(folder):
  '''Every reach upstream of the outlet reach nearest to MOUTH, in breadth-first order from the outlet reach.'''
  base = next(os.path.join(folder, f[:-4]) for f in sorted(os.listdir(folder)) if f.endswith('.shp'))
  cols, names = read_dbf(base + '.dbf', FIELDS)
  print('fields:', names)
  ids = cols['HYRIV_ID'].astype(np.int64)
  nd = cols['NEXT_DOWN'].astype(np.int64)
  n = len(ids)
  offs, _ = read_shx(base + '.shx')
  assert len(offs) == n
  outlets = np.flatnonzero(nd == 0)
  print('reaches in the file:', n, '| outlet reaches:', len(outlets))
  with open(base + '.shp', 'rb') as fh:
    best = None
    for i in outlets:
      p = read_polyline(fh, offs[i])[-1]  # downstream end of the outlet reach
      dist = np.hypot((p[0] - MOUTH[0]) * np.cos(np.radians(MOUTH[1])), p[1] - MOUTH[1])
      if best is None or dist < best[0]:
        best = (dist, i, p)
    dist, i0, p0 = best
    print(f'nearest outlet reach: HYRIV_ID {ids[i0]}, {dist * 111:.1f} km from the mouth', end=' ')
    print(f'| UPLAND_SKM {cols["UPLAND_SKM"][i0]:.0f}')
    parents = {}
    for k in range(n):
      parents.setdefault(nd[k], []).append(k)
    basin, frontier = [i0], [ids[i0]]
    while frontier:
      nxt = []
      for rid in frontier:
        for k in parents.get(rid, []):
          basin.append(k)
          nxt.append(ids[k])
      frontier = nxt
    lines = {int(ids[k]): read_polyline(fh, offs[k]).tolist() for k in basin}
  print('reaches in the basin:', len(basin), '| on the main river:', int(np.sum(cols['MAIN_RIV'][basin] == ids[i0])))
  rows = [
    dict(
      id=int(ids[k]),
      next_down=int(nd[k]),
      length_km=float(cols['LENGTH_KM'][k]),
      upland_skm=float(cols['UPLAND_SKM'][k]),
      catch_skm=float(cols['CATCH_SKM'][k]),
      order=int(cols['ORD_STRA'][k]),
      discharge_cms=float(cols['DIS_AV_CMS'][k]),
      start=lines[int(ids[k])][0],
      end=lines[int(ids[k])][-1],
    )
    for k in basin
  ]
  orders = [x['order'] for x in rows]
  print('stream orders:', {o: orders.count(o) for o in sorted(set(orders))}, end=' ')
  print('| total length km:', round(sum(x['length_km'] for x in rows), 1))
  return dict(source=SOURCE, outlet=int(ids[i0]), mouth=MOUTH, reaches=rows, lines=lines)


def network(basin):
  '''The population structure on the reaches of the basin, with vertex v the reach basin['reaches'][v].

  Each reach after the outlet reach adds its downstream edge DOWN times and its upstream edge UP times.
  The temperature of a vertex is the sum over its neighbors of the probability that an offspring of the neighbor is
  placed at the vertex.
  '''
  rows = basin['reaches']
  assert rows[0]['id'] == basin['outlet'] and rows[0]['next_down'] == 0
  ids = [x['id'] for x in rows]
  index = {rid: v for v, rid in enumerate(ids)}
  edges = []
  for v, x in enumerate(rows[1:], start=1):
    p = index[x['next_down']]
    assert p < v
    edges += [[v, p] for _ in range(DOWN)] + [[p, v] for _ in range(UP)]
  N = len(ids)
  W = np.zeros((N, N))
  for a, b in edges:
    W[a, b] += 1
  theta = (W / W.sum(axis=1, keepdims=True)).sum(axis=0)
  return dict(
    N=N,
    q_up=UP / DOWN,
    ids=ids,
    edges=edges,
    theta=theta.tolist(),
    order=[x['order'] for x in rows],
    upland_skm=[x['upland_skm'] for x in rows],
    outlet=0,
  )


if __name__ == '__main__':
  parts = sys.argv[1:]
  if parts and parts[0] == 'extract':
    with open(path(SECTION, 'charles_basin.json'), 'w') as f:
      json.dump(extract(parts[1]), f)
  net = network(load(SECTION, 'charles_basin.json'))
  with open(path(SECTION, 'charles_network.json'), 'w') as f:
    json.dump(net, f)
  write_edges(net['N'], net['edges'], path(SECTION, 'charles_net.txt'))
  print('wrote', net['N'], 'vertices and', len(net['edges']), 'directed edges')
