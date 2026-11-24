'''Generate the certified numerical bounds for C = sum_{l>=1} sqrt(l) e^(1 - sqrt(l)) < 10.6.

For l = 1..30: rational lo <= sqrt(l) <= hi (4 decimals), lo = n + delta, and a decimal d with
hi * E_HI <= d * E_LO^n * T8(delta), where T8 is the Taylor polynomial of exp of degree 7,
E_LO = 2.71828 < e and E_HI = 2.7182819 > e. Then f(l) <= d.
Tail: G(30) = 2 e^(1 - sqrt 30) (30 + 2 sqrt 30 + 2) <= d_tail.
All checks are exact (fractions). Prints the Lean block.
'''

from fractions import Fraction as F
from math import isqrt, factorial, sqrt, exp

E_LO = F(271828, 100000)
E_HI = F(27182819, 10000000)


def T8(d):
  return sum(d**i / factorial(i) for i in range(8))


def ceil_places(fr, places):
  s = 10**places
  return F(-((-fr.numerator * s) // fr.denominator), s)


def dec(fr, places):
  s = 10**places
  v = fr * s
  assert v.denominator == 1, (fr, places)
  v = v.numerator
  if places == 0:
    return str(v)
  ip, fp = divmod(v, s)
  out = f'{ip}.{fp:0{places}d}'.rstrip('0').rstrip('.')
  return out


def split(l):
  r = isqrt(l)
  if r * r == l:
    lo = hi = F(r)
  else:
    s = isqrt(l * 10**8)
    lo = F(s, 10**4)
    hi = F(s + 1, 10**4)
  assert lo * lo <= l <= hi * hi
  n = int(lo)
  delta = lo - n
  return lo, hi, n, delta


lines = []
total = F(0)
for l in range(1, 31):
  lo, hi, n, delta = split(l)
  bound = hi * E_HI / (E_LO**n * T8(delta))
  d = ceil_places(bound, 5)
  assert hi * E_HI <= d * (E_LO**n * T8(delta))
  true = sqrt(l) * exp(1 - sqrt(l))
  assert true <= float(d) < true + 0.002, (l, true, float(d))
  total += d
  lines.append(
    f'lemma fC_le_{l} : fC {l} ≤ {dec(d, 5)} :=\n'
    f'  term_cert {l} {dec(lo, 4)} {dec(hi, 4)} {dec(delta, 4)} {dec(d, 5)} {n} (by norm_num) (by norm_num)\n'
    f'    (by norm_num) (by norm_num) (by norm_num) (by norm_num)\n'
    f'    (by norm_num [Finset.sum_range_succ, Nat.factorial])\n'
  )

lo, hi, n, delta = split(30)
tb = 2 * E_HI * (30 + 2 * hi + 2) / (E_LO**n * T8(delta))
dt = ceil_places(tb, 5)
assert 2 * E_HI * (30 + 2 * hi + 2) <= dt * (E_LO**n * T8(delta))
lines.append(
  f'lemma Gt_le_30 : Gt 30 ≤ {dec(dt, 5)} :=\n'
  f'  Gt_cert 30 {dec(lo, 4)} {dec(hi, 4)} {dec(delta, 4)} {dec(dt, 5)} {n} (by norm_num) (by norm_num)\n'
  f'    (by norm_num) (by norm_num) (by norm_num) (by norm_num)\n'
  f'    (by norm_num [Finset.sum_range_succ, Nat.factorial])\n'
)
print('\n'.join(lines))
print(f'-- S30 = {dec(total, 5)}  T30 = {dec(dt, 5)}  sum = {float(total + dt):.5f}')
assert total + dt < F(106, 10)
true30 = sum(sqrt(l) * exp(1 - sqrt(l)) for l in range(1, 31))
print(f'-- true partial sum = {true30:.6f}')
