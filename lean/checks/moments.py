# Compact closed forms of E[H | fix] and E[H^2 | fix] in q = 1/r, Q = q^N, n = N.
import sympy as sp

q, Q, n = sp.symbols('q Q n')
A = (
  n
  * (
    n * Q**2 * q**3
    - n * Q**2 * q**2
    + n * Q * q**3
    + 3 * n * Q * q**2
    - 3 * n * Q * q
    - n * Q
    + n * q
    - n
    + Q**2 * q**2
    + Q**2 * q
    - q**2
    - q
  )
  / ((Q - 1) ** 3 * (q - 1) ** 3)
)
B = (2 * n * Q * q**2 - 2 * n * Q * q + 2 * n * q - 2 * n + Q * q**2 + Q * q - q**2 - q) / (
  (Q - 1) ** 2 * (q - 1) ** 3
)
C = (
  q
  * (2 * n * Q * q**2 - 2 * n * Q * q + 2 * n * q - 2 * n + Q * q + Q - q - 1)
  / ((Q - 1) ** 2 * (q - 1) ** 3)
)


def Wf(t, x):
  return (
    -A
    + A * t
    - B * x
    - C * x * t
    - x**2 / ((Q - 1) * (q - 1) ** 2)
    + q**2 * x**2 * t / ((Q - 1) * (q - 1) ** 2)
  )


def Uf(t, x):
  return (n * (1 - t) * (1 + q * Q) - x * (1 - Q) * (1 + q * t)) / ((1 - q) * (1 - Q) ** 2)


h1 = (1 - q) / (1 - Q)
EH2 = sp.factor(sp.cancel(Wf(q, 1) / h1))
EH = sp.factor(sp.cancel(Uf(q, 1) / h1))
print('EH  =', EH)
print('EH2 =', EH2)
num2, den2 = sp.fraction(sp.together(EH2))
num1, den1 = sp.fraction(sp.together(EH))
print('EH2 num =', sp.expand(num2))
print('EH2 den =', den2)
print('EH num =', sp.expand(num1))
print('EH den =', den1)
lead = q * (1 + q) / (1 - q) ** 3 * (n - 2 * (1 + q) / (1 - q))
rem = n * Q * (1 + q) * (n * (1 - q**2) + 2 * q * (1 - Q)) / ((1 - Q) ** 2 * (1 - q) ** 3)
print('check var identity:', sp.simplify(EH2 - EH**2 - lead - rem))
