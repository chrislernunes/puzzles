#!/usr/bin/env python3
"""
Jane Street August 2026 — Andy's Afternoon Amble

Andy lives on the 4 white hexagons of a truncated tetrahedron (graph = K4:
every white hex touches the other three). He marks home, walks to a random
white neighbour at each step, and stops at first return to home.

He fell onto the kitchen floor: an infinite hex tiling in which white hexes
have 3 white neighbours (alternating black/white around each white) and
black hexes have 6 white neighbours. Locally this is identical — 3 white
exits, 120° apart — so he uses the same amble. He remembers every turn
(Back / Left / Right).

He discovers the floor iff the turn-sequence, replayed on K4, would have
returned to "home" at a different moment than the pheromone-marked floor
hex. Equivalently: there is a turn-preserving covering map

    ψ : (white floor hexes) → K4

and he is fooled iff he returns to the real home before hitting any other
hex in the fibre ψ⁻¹(home).

This script derives p = P(discovers) exactly. Nothing about 11/20 is
hard-coded; it falls out of a 4-state linear system whose states are
read off the covering, then is cross-checked by Monte Carlo and by a
finite-grid rational solve.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from random import Random

import sympy as sp


# ---------------------------------------------------------------------------
# 1. The ball: K4 with a rotation system
# ---------------------------------------------------------------------------
# rho[v] = CCW cyclic order of the 3 neighbours of v.
# This orientation makes every face a triangle (the black triangles).
rho = {0: [1, 2, 3], 1: [0, 3, 2], 2: [0, 1, 3], 3: [0, 2, 1]}


def k4_move(dart, turn):
    u, v = dart
    if turn == "B":
        return (v, u)
    nb = rho[v]
    i = nb.index(u)
    return (v, nb[(i + 1) % 3]) if turn == "L" else (v, nb[(i - 1) % 3])


def faces_are_triangles():
    darts = [(u, v) for u in range(4) for v in range(4) if u != v]

    def left(d):
        u, v = d
        nb = rho[v]
        return (v, nb[(nb.index(u) + 1) % 3])

    for d in darts:
        x = d
        for _ in range(3):
            x = left(x)
        if x != d:
            return False
    return True


assert faces_are_triangles()


# ---------------------------------------------------------------------------
# 2. The floor: white hexes of the 3-coloured hex tiling
# ---------------------------------------------------------------------------
# Axial coordinates (q, r). Colour (q - r) mod 3:
#   0 = black (6 white neighbours)
#   1, 2 = white (3 white neighbours, alternating with black)
def color(v):
    return (v[0] - v[1]) % 3


# CCW white-neighbour offsets, one list per white colour.
W1 = [(1, 0), (0, -1), (-1, 1)]
W2 = [(1, -1), (-1, 0), (0, 1)]


def wnbrs(v):
    ds = W1 if color(v) == 1 else W2
    return [(v[0] + dq, v[1] + dr) for dq, dr in ds]


def hc_move(dart, turn):
    u, v = dart
    if turn == "B":
        return (v, u)
    nb = wnbrs(v)
    i = nb.index(u)
    return (v, nb[(i + 1) % 3]) if turn == "L" else (v, nb[(i - 1) % 3])


HOME = (1, 0)
assert color(HOME) == 1


# ---------------------------------------------------------------------------
# 3. Covering map ψ by propagating turns
# ---------------------------------------------------------------------------
def apply_word(dart, word, mover):
    for t in word:
        dart = mover(dart, t)
    return dart


# Sanity: ball relations B² = L³ = (LB)³ = id; floor faces are 6-cycles.
darts_k4 = [(u, v) for u in range(4) for v in range(4) if u != v]
assert all(apply_word(d, "BB", k4_move) == d for d in darts_k4)
assert all(apply_word(d, "LLL", k4_move) == d for d in darts_k4)
assert all(apply_word(d, "LBLBLB", k4_move) == d for d in darts_k4)
d0 = (wnbrs(HOME)[0], HOME)
assert apply_word(d0, "LLLLLL", hc_move) == d0
assert apply_word(d0, "LBLBLB", hc_move) == d0


def build_covering(R=24):
    """Dart-level covering on the axial hex of radius R, then descend to vertices."""

    def inpatch(v):
        return max(abs(v[0]), abs(v[1]), abs(v[0] + v[1])) <= R

    start_floor = (wnbrs(HOME)[0], HOME)  # dart into HOME
    phi = {start_floor: (1, 0)}  # image dart into ball-vertex 0
    Q = deque([start_floor])
    conflicts = 0
    while Q:
        d = Q.popleft()
        for t in "BLR":
            nd = hc_move(d, t)
            if not (inpatch(nd[0]) and inpatch(nd[1])):
                continue
            img = k4_move(phi[d], t)
            if nd in phi:
                if phi[nd] != img:
                    conflicts += 1
            else:
                phi[nd] = img
                Q.append(nd)

    psi = {}
    for (u, v), (a, b) in phi.items():
        if v in psi:
            assert psi[v] == b
        else:
            psi[v] = b
    assert conflicts == 0
    # Rainbow: every fully-interior white sees the three *other* K4 labels.
    for v, lab in psi.items():
        ns = wnbrs(v)
        if all(n in psi for n in ns):
            assert sorted(psi[n] for n in ns) == sorted({0, 1, 2, 3} - {lab})
    return psi


psi = build_covering()
fiber = [v for v, lab in psi.items() if lab == 0]
print(f"covering: {len(psi)} white hexes, {len(fiber)} in the home fibre "
      f"({len(fiber) / len(psi):.3f} ≈ 1/4)")


# ---------------------------------------------------------------------------
# 4. The 6-cycle after the first step, and the 4-state system
# ---------------------------------------------------------------------------
# After one step Andy sits on a neighbour A0 of HOME. The two other
# neighbours of A0, together with A0, lie on a unique 6-cycle C that does
# *not* contain HOME:
#
#        2 --- 3 --- 2
#       /             \
#      1               1
#       \             /
#        0 ---·--- (A0 = 0)
#              \
#             HOME
#
# Labels are graph-distance along C from A0. Leaving C:
#   • from 0, the off-cycle neighbour is HOME          → fooled (safe return)
#   • from 1, 2 or 3, the off-cycle neighbour is an
#     impostor (ψ = 0, not HOME)                       → discovers
#
# Let q_i = P(return to HOME before any impostor | now at a vertex labelled i).
# Then q_i is exactly the JS "p_i", and P(fooled) = q_0 because the amble
# is already at A0 after the forced first step. P(discovers) = 1 - q_0.

A0 = wnbrs(HOME)[0]
off_home = [n for n in wnbrs(A0) if n != HOME]
assert len(off_home) == 2

# The honeycomb face at A0 that does *not* contain HOME. Try both
# off-home incoming darts and both turn directions; exactly one 6-cycle
# uses both non-home edges.
C = None
for src in off_home:
    for turn in "LR":
        cyc = []
        prev, cur = src, A0
        for _ in range(6):
            cyc.append(cur)
            prev, cur = cur, hc_move((prev, cur), turn)[1]
        if HOME not in cyc and set(off_home) <= set(cyc) and len(set(cyc)) == 6:
            C = cyc
            break
    if C is not None:
        break
assert C is not None
print("6-cycle after first step:", C)

# Label by cycle-distance from A0 (min of both ways) → {0,1,2,3,2,1}
def cyc_label(v):
    i = C.index(v)
    return min(i, 6 - i)


labels = {v: cyc_label(v) for v in C}
print("cycle labels:", {v: labels[v] for v in C})

# Confirm spokes off C: from 0 it's HOME; from 1,2,3 it's an impostor.
for v in C:
    off = [n for n in wnbrs(v) if n not in C]
    assert len(off) == 1
    o = off[0]
    if labels[v] == 0:
        assert o == HOME
    else:
        assert o in psi and psi[o] == 0 and o != HOME, (v, o, psi.get(o))
print("off-cycle spokes: HOME at label 0, impostors at labels 1,2,3")


# q0 = 1/3 + 2/3 q1          (1/3 → HOME safe, 2/3 → label 1)
# q1 = 1/3 q0 + 1/3 q2       (1/3 → 0, 1/3 → 2, 1/3 → impostor)
# q2 = 1/3 q1 + 1/3 q3       (1/3 → 1, 1/3 → 3, 1/3 → impostor)
# q3 = 2/3 q2                (2/3 → 2, 1/3 → impostor)

q0, q1, q2, q3 = sp.symbols("q0 q1 q2 q3")
eqs = [
    sp.Eq(q0, sp.Rational(1, 3) + sp.Rational(2, 3) * q1),
    sp.Eq(q1, sp.Rational(1, 3) * q0 + sp.Rational(1, 3) * q2),
    sp.Eq(q2, sp.Rational(1, 3) * q1 + sp.Rational(1, 3) * q3),
    sp.Eq(q3, sp.Rational(2, 3) * q2),
]
sol = sp.solve(eqs, [q0, q1, q2, q3], dict=True)[0]
print("\nExact state values:")
for s in (q0, q1, q2, q3):
    print(f"  {s} = {sol[s]}")

fooled = sol[q0]
p = 1 - fooled
print(f"\nP(fooled)     = {fooled}")
print(f"P(discovers)  = {p}   = {sp.fraction(p)}")
assert p == sp.Rational(11, 20)


# ---------------------------------------------------------------------------
# 5. Independent check: absorbing system on a finite patch (rationals)
# ---------------------------------------------------------------------------
def patch_hitting_fraction(R=8):
    """
    States = white hexes in the axial hex of radius R with ψ ≠ 0.
    Absorbing: real HOME (value 0 = fooled) and impostors / boundary
    (value 1 = discovered). Solve (I - P) x = b over Q.
    """
    whites = [
        (q, r)
        for q in range(-R, R + 1)
        for r in range(-R, R + 1)
        if max(abs(q), abs(r), abs(q + r)) <= R and color((q, r)) != 0
    ]
    vset = set(whites)
    transient = [v for v in whites if psi.get(v, 0) != 0]
    idx = {v: i for i, v in enumerate(transient)}
    n = len(transient)
    A = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    b = [Fraction(0) for _ in range(n)]
    for v in transient:
        i = idx[v]
        A[i][i] = 1
        for u in wnbrs(v):
            if u not in vset:
                b[i] += Fraction(1, 3)  # boundary: count as discovered (upper bd)
            elif psi.get(u, 0) == 0:
                if u != HOME:
                    b[i] += Fraction(1, 3)  # impostor
                # else: real home → contribution 0 (fooled)
            else:
                A[i][idx[u]] -= Fraction(1, 3)

    # Gaussian elimination over Q
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        fac = M[col][col]
        M[col] = [x / fac for x in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f:
                M[r] = [a - f * c for a, c in zip(M[r], M[col])]
    hit = [M[i][n] for i in range(n)]
    # First step is uniform onto the 3 neighbours of HOME; those are transient.
    return sum(hit[idx[u]] for u in wnbrs(HOME)) / 3


p_patch = patch_hitting_fraction(8)
print(f"\nFinite-patch upper bound (R=8, rational): {p_patch} = {float(p_patch):.10f}")
assert p_patch == Fraction(11, 20), p_patch


# ---------------------------------------------------------------------------
# 6. Monte Carlo on the coupled (floor, ball) walk
# ---------------------------------------------------------------------------
rng = Random(7)
N = 200_000
discoveries = 0
for _ in range(N):
    k = rng.randrange(3)
    pd = (HOME, wnbrs(HOME)[k])
    sd = (0, rho[0][k])
    while True:
        shadow_home = sd[1] == 0
        real_home = pd[1] == HOME
        if shadow_home or real_home:
            if not (shadow_home and real_home):
                discoveries += 1
            break
        t = rng.choice("BLR")
        pd = hc_move(pd, t)
        sd = k4_move(sd, t)

mc = discoveries / N
print(f"Monte Carlo ({N} walks): p ≈ {mc:.4f}   (11/20 = {11/20:.4f})")

print("\nANSWER:", p)
