'''Figure 4: the directed cycles of panels b to d and the structure test of panels e to g.

Run: python -m fixloc.make.fig4 (all panels), or python -m fixloc.make.fig4 cycles (panels b to d only, a few seconds).
'''
from multiprocessing import Pool

import numpy as np

from fixloc import structure as st
from fixloc.exact import conditioned
from fixloc.make import dump
from fixloc.walk import exact_summary

R = 2.0
N = 7
TRUE = [(0, 1), (0, 4), (1, 4), (2, 3), (3, 5), (4, 5), (5, 6)]
THEO = [(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (4, 5), (5, 6)]
SEARCH_PAIR = [3, 6]
MONITORED = (2, 3)


def pair(sequences=40000, nmax=30, seed=2026):
  '''Exact summaries of the two structures and simulated probabilities of choosing the true structure.

  SEARCH_PAIR is the best monitored pair of the pair search, with the other vertices as one observation.
  '''
  eg, eh = st.summary(N, TRUE, R), st.summary(N, THEO, R)
  cat = lambda P: np.r_[P[SEARCH_PAIR], 1 - P[SEARCH_PAIR].sum()]
  rng = np.random.default_rng(seed)
  a_mon = st.simulated_accuracy(cat(eg['P']), cat(eh['P']), rng, sequences, nmax)
  a_all = st.simulated_accuracy(eg['P'], eh['P'], rng, sequences, nmax)
  a_time = st.simulated_accuracy(eg['pmf'], eh['pmf'], rng, sequences, nmax)
  sample = rng.choice(3, size=20, p=cat(eg['P']))
  return dict(
    r=R,
    N=N,
    G=TRUE,
    H=THEO,
    moved=dict(removed=[e for e in TRUE if e not in THEO], added=[e for e in THEO if e not in TRUE]),
    monitored=SEARCH_PAIR,
    rho=[eg['rho'], eh['rho']],
    T=[eg['T'], eh['T']],
    PG=eg['P'].tolist(),
    PH=eh['P'].tolist(),
    thetaG=eg['theta'].tolist(),
    thetaH=eh['theta'].tolist(),
    cat_G=cat(eg['P']).tolist(),
    cat_H=cat(eh['P']).tolist(),
    n=list(range(1, nmax + 1)),
    acc_monitored=a_mon.tolist(),
    acc_all=a_all.tolist(),
    acc_time=a_time.tolist(),
    sample=sample.tolist(),
    sequences=sequences,
    seed=seed,
  )


def observable(seed=11):
  '''Exact probabilities of the observation at the monitored vertices and 20 simulated fixation events.'''
  cG, cH = st.chain(N, TRUE, R), st.chain(N, THEO, R)
  PTG, PTH = conditioned(cG), conditioned(cH)
  wG, wH = cG.rho / cG.rho.sum(), cH.rho / cH.rho.sum()
  rG = st.last_occupied(PTG, wG, cG.full, [MONITORED])[0]
  rH = st.last_occupied(PTH, wH, cH.full, [MONITORED])[0]
  acc = st.binomial_accuracy(rG[0], rH[0])
  first = next(i + 1 for i, a in enumerate(acc) if a >= 0.95)
  stays = max(i + 2 for i, a in enumerate(acc) if a < 0.95) if any(a < 0.95 for a in acc) else 1
  ev = st.sample_events(PTG, wG, cG.full, MONITORED, seed)
  rhoG, rhoH = float(cG.rho.mean()), float(cH.rho.mean())
  nf = np.unique(np.round(np.logspace(0, 7, 141)).astype(int)).tolist()
  return dict(
    monitored=list(MONITORED),
    shapes={'2': 4, '3': 3},
    P_true=rG.tolist(),
    P_theorized=rH.tolist(),
    n=list(range(1, len(acc) + 1)),
    acc_two_sites=acc,
    first_n_095=first,
    stays_above_095_from=stays,
    seed=seed,
    events=ev,
    counts={str(v): sum(e['record'] == v for e in ev) for v in MONITORED},
    rho=[rhoG, rhoH],
    n_fixation=nf,
    acc_fixation=st.outcome_accuracy(rhoG, rhoH, nf),
    first_n_095_fixation=st.first_outcome_count(rhoG, rhoH),
  )


def time_accuracy(blocks=32, m=125, seed=2026):
  '''Probability of choosing the true structure from fixation times, from blocks * m simulated sequences.'''
  cG, cH = st.chain(N, TRUE, R), st.chain(N, THEO, R)
  pg, ph = st.time_pmf(cG, conditioned(cG)), st.time_pmf(cH, conditioned(cH))
  llr = np.log(np.maximum(pg, 1e-300)) - np.log(np.maximum(ph, 1e-300))
  cdf = np.cumsum(pg / pg.sum())
  ck = np.unique(np.round(np.logspace(0, 5, 41)).astype(int))
  with Pool() as pool:
    res = pool.starmap(st.time_block, [(llr, cdf, ck, seed + i, m) for i in range(blocks)])
  allv = np.concatenate(res, axis=1)
  acc = ((allv > 0) + 0.5 * (allv == 0)).mean(1)
  kl = float((pg * llr).sum())
  var = float((pg * llr**2).sum() - kl**2)
  return dict(
    n=ck.tolist(), acc_time=acc.tolist(), sequences=int(allv.shape[1]), kl_per_event=kl, var_per_event=var
  )


def compute():
  return dict(pair=pair(), observable=observable(), time_accuracy=time_accuracy())


def write(res):
  q, ob, ta = res['pair'], res['observable'], res['time_accuracy']
  e = {k: q[k] for k in ('r', 'N', 'G', 'H', 'moved', 'PG', 'PH', 'rho', 'T')}
  dump('fig4', 'e', dict(**e, events=ob['events'][:5]))
  dump('fig4', 'f', {k: ob[k] for k in ('monitored', 'P_true', 'P_theorized', 'counts')})
  g = dict(n_two_sites=ob['n'], acc_two_sites=ob['acc_two_sites'], n_all=q['n'], acc_all=q['acc_all'])
  g.update(n_time=ta['n'], acc_time=ta['acc_time'])
  g.update(n_fixation=ob['n_fixation'], acc_fixation=ob['acc_fixation'])
  dump('fig4', 'g', g)


CYCLE_N = (5, 10, 20, 30, 40)
CYCLE_R = (1, 2, 3)


def cycles():
  '''Exact fixation location probabilities on the directed cycle, from a single initial mutant at vertex 1, given fixation.'''
  return {r: [exact_summary(n, r)['offset_prob'] for n in CYCLE_N] for r in CYCLE_R}


def write_cycles(res):
  for panel, r in zip('bcd', CYCLE_R):
    dump('fig4', panel, dict(r=r, N=list(CYCLE_N), P=res[r]))


if __name__ == '__main__':
  import sys

  if sys.argv[1:] != ['cycles']:
    write(compute())
  write_cycles(cycles())
