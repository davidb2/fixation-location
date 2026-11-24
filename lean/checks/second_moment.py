# Exact second moment of the number H of head advances on the directed cycle, on the event of fixation.
import sympy as sp

q, x, N = sp.symbols('q x N', positive=True)
Qx, QN = sp.symbols('Qx QN', positive=True)  # q**x and q**N as atoms
a = 1 / (1 + q)  # alpha = r/(1+r) with r = 1/q
b = q / (1 + q)  # beta = 1/(1+r)
h = lambda X, QX: (1 - QX) / (1 - QN)
U = lambda X, QX: (N * (1 - QX) * (1 + q * QN) - X * (1 - QN) * (1 + q * QX)) / ((1 - q) * (1 - QN) ** 2)
# check U against its first-step equation
eqU = U(x, Qx) - (a * (U(x + 1, q * Qx) + h(x + 1, q * Qx)) + b * U(x - 1, Qx / q))
print('U equation residual:', sp.simplify(eqU))
c = sp.symbols('c0:6')
W = lambda X, QX: c[0] + c[1] * QX + c[2] * X + c[3] * X * QX + c[4] * X**2 + c[5] * X**2 * QX
res = sp.expand(
  sp.together(
    W(x, Qx) - (a * (W(x + 1, q * Qx) + 2 * U(x + 1, q * Qx) + h(x + 1, q * Qx)) + b * W(x - 1, Qx / q))
  )
  * (1 + q)
  * (1 - q)
  * (1 - QN) ** 2
)
poly = sp.Poly(res, x, Qx)
eqs = [sp.Eq(co, 0) for co in poly.coeffs()]
eqs += [sp.Eq(W(0, 1), 0), sp.Eq(W(N, QN), 0)]
sol = sp.solve(eqs, c, dict=True)
print('solutions:', len(sol))
s0 = sol[0]
Wsol = {k: sp.factor(sp.simplify(v)) for k, v in s0.items()}
for k in c:
  print(k, '=', Wsol.get(k, k))
W1 = sp.simplify(W(1, q).subs(Wsol))
h1 = h(1, q)
U1 = U(1, q)
EH2 = sp.simplify(W1 / h1)
EH = sp.simplify(U1 / h1)
var = sp.simplify(EH2 - EH**2)
r = sp.symbols('r', positive=True)
var_r = sp.simplify(var.subs(q, 1 / r))
lead = r * (r + 1) / (r - 1) ** 3 * (N - 2 * (r + 1) / (r - 1))
rem = sp.simplify(sp.together(var_r - lead))
print('Var exact (r form):', sp.factor(var_r))
print('remainder:', sp.factor(rem))
