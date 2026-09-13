#!/usr/bin/env python3
"""
Jane Street July 2026 — 'Pent-Up' Frustration 3 / Knight Moves 7

Reconstruct the knight's path from the published board (regions + written
scores) and compute the required neighbour-sum over unvisited squares.

Nothing about the path, K, move types, tower locations, or the numeric
answer is hard-coded. Those are all derived.

Board coordinates: row 0 is the top of the published image, col 0 is left.
Start square is bottom-left = (7, 0), shown with score 0.
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict

sys.setrecursionlimit(100_000)

# ---------------------------------------------------------------------------
# 1. Board as printed (thick region borders + written scores)
# ---------------------------------------------------------------------------
# 13 regions: 12 pentominoes + one 2x2 tetromino. Labels are arbitrary;
# they just group squares that share an interior.
REG = [
    "AAAAABBB",
    "CCCDDEEB",
    "CFCDDDEB",
    "GFFHHIEE",
    "GGFHHIIJ",
    "GKFLIIMJ",
    "GKLLLMMJ",
    "KKKLMMJJ",
]

region: dict[tuple[int, int], str] = {}
cells_of: dict[str, list[tuple[int, int]]] = defaultdict(list)
for r in range(8):
    for c in range(8):
        g = REG[r][c]
        region[(r, c)] = g
        cells_of[g].append((r, c))

assert sorted(len(v) for v in cells_of.values()) == [4] + [5] * 12

# Written scores on the puzzle image (start square 0 is listed separately).
NUM = {
    (0, 5): 37,
    (0, 7): 1100,
    (2, 3): 23,
    (2, 5): 138,
    (3, 0): 528,
    (4, 1): 449,
    (4, 4): 16,
    (5, 1): 750,
    (5, 3): 88,
    (5, 5): 272,
    (5, 6): 1,
}
START = (7, 0)
SQ_OF_VAL = {v: p for p, v in NUM.items()}
VALS = set(NUM.values())

# 3D knight: permute (0, 1, 2). Heights are only 1 or 2, so Δz ∈ {0, ±1}.
#   same height -> classic (1,2) knight
#   up / down   -> (0,±2) or (±2,0) rook-2 slide
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
UPDOWN = [(0, 2), (0, -2), (2, 0), (-2, 0)]
inb = lambda p: 0 <= p[0] < 8 and 0 <= p[1] < 8

CAP = 10**9  # scores on the board are ≤ 1100; this just stops runaway multiplies


# ---------------------------------------------------------------------------
# 2. Score-space forcing (ignore the board)
# ---------------------------------------------------------------------------
def reach(states: set[tuple[int, int]], t0: int, t1: int) -> set[tuple[int, int]]:
    """All (score, height) reachable from `states` after moves t0+1 .. t1."""
    for N in range(t0 + 1, t1 + 1):
        nxt: set[tuple[int, int]] = set()
        for v, h in states:
            nxt.add((v + N, h))
            if h == 1 and v * N <= CAP:
                nxt.add((v * N, 2))
            if h == 2 and v % N == 0:
                nxt.add((v // N, 1))
        states = nxt
    return states


def op_paths(s: int, h: int, t0: int, t1: int, goal: int):
    """Every operator sequence taking (s, h) at t0 to score `goal` at t1."""
    out = []

    def rec(v, hh, N, ops):
        if N > t1:
            if v == goal:
                out.append((ops, hh))
            return
        if v + N <= CAP:
            rec(v + N, hh, N + 1, ops + [f"+{N}"])
        if hh == 1 and v * N <= CAP:
            rec(v * N, 2, N + 1, ops + [f"*{N}"])
        if hh == 2 and v % N == 0:
            rec(v // N, 1, N + 1, ops + [f"/{N}"])

    rec(s, h, t0 + 1, [])
    return out


print("=== Early checkpoints (moves 3,6,9,12,15,18) ===")
states = {(0, 1), (0, 2)}  # start height unknown a priori
t, used_vals = 0, []
for cp in (3, 6, 9, 12, 15, 18):
    states = reach(states, t, cp)
    hits = sorted({v for v, _h in states if v in VALS and v not in used_vals})
    assert len(hits) == 1, f"ambiguous / empty checkpoint at move {cp}: {hits}"
    v = hits[0]
    used_vals.append(v)
    states = {(vv, hh) for vv, hh in states if vv == v}
    print(f"  move {cp:2d}: score must be {v:4d}  at {SQ_OF_VAL[v]}")
    t = cp
print("forced early values:", used_vals)

LATE = VALS - set(used_vals)
print("\n=== Which K can reach a leftover written score from 88 in K moves? ===")
viable_K = []
for K in range(4, 10):
    hits = sorted({v for v, _h in reach({(88, 1)}, 18, 18 + K) if v in LATE})
    print(f"  K = {K}: reachable -> {hits}")
    if hits:
        viable_K.append(K)

print("\n=== Viable late-checkpoint orders ===")
forced_K_order = []
for K in viable_K:
    for order in itertools.permutations(sorted(LATE)):
        st, tt, ok = {(88, 1)}, 18, True
        for goal in order:
            st = {(v, h) for v, h in reach(st, tt, tt + K) if v == goal}
            tt += K
            if not st:
                ok = False
                break
        if ok:
            print(f"  K = {K}, order = {order}")
            forced_K_order.append((K, order))

assert forced_K_order, "no (K, order) survives score-space"
print()
for K, order in forced_K_order:
    print(f"Unique-looking op sequences for K = {K}:")
    s, h, t = 88, 1, 18
    unique = True
    for goal in order:
        paths = op_paths(s, h, t, t + K, goal)
        if len(paths) != 1:
            unique = False
        print(f"  moves {t + 1}-{t + K}: {len(paths)} sequence(s)",
              f"-> {goal}" + (f"  {paths[0][0]}" if len(paths) == 1 else ""))
        if not paths:
            unique = False
            break
        s, h, t = goal, paths[0][1], t + K
    print("  (sequences unique)" if unique else "  (multiple sequences)")


# ---------------------------------------------------------------------------
# 3. Board search
# ---------------------------------------------------------------------------
# Towers are chosen on the fly: landing at height 2 on a square claims that
# square as its region's unique tower. A region that is fully visited with
# no tower is dead. Checkpoints must land on an unused numbered square whose
# value equals the new score; non-checkpoint moves may not land on a numbered
# square (those numbers were only written on checkpoint arrivals).
#
# The tour ends the instant the 13th tower is visited, and that instant must
# fall in [18 + 5K, 18 + 6K - 1] so that exactly the five leftover numbers
# are the post-18 writings.

solutions = []


def solve_for_K(K: int) -> None:
    cps = [3, 6, 9, 12, 15, 18] + [18 + i * K for i in range(1, 6)]
    cpset = set(cps)
    Lmin, Lmax = 18 + 5 * K, 18 + 6 * K - 1
    VALS_sorted = sorted(VALS)
    feas_memo: dict = {}

    def next_cp(t):
        for c in cps:
            if c > t:
                return c
        return None

    def feas(t, score, h, used_fs) -> bool:
        """Can the remaining written scores still be hit on schedule?"""
        nc = next_cp(t)
        if nc is None:
            return True
        key = (t, score, h, used_fs)
        if key in feas_memo:
            return feas_memo[key]
        ok = False
        N = t + 1
        if score <= CAP:
            targets = [v for v in VALS_sorted if v not in used_fs]

            def arrive(s2, h2):
                return (s2 in targets) if N == nc else feas(N, s2, h2, used_fs)

            if arrive(score + N, h):
                ok = True
            if not ok and h == 1 and score * N <= CAP and arrive(score * N, 2):
                ok = True
            if not ok and h == 2 and score % N == 0 and arrive(score // N, 1):
                ok = True
        feas_memo[key] = ok
        return ok

    heights: dict[tuple[int, int], int] = {}
    towers: dict[str, tuple[int, int]] = {}
    used: set[int] = set()
    path = [START]
    scores = [0]
    visited = {START}

    def dfs(t, pos, score):
        if len(towers) == 13:
            if Lmin <= t <= Lmax:
                solutions.append((K, list(path), list(scores), dict(towers)))
            return
        if t >= Lmax or 13 - len(towers) > Lmax - t:
            return

        N = t + 1
        h = heights[pos]
        is_cp = N in cpset
        nc = next_cp(t)
        if nc is not None:
            rem = nc - t
            cands = [SQ_OF_VAL[v] for v in VALS if v not in used]
            # each move covers at most 2 rows and 2 cols
            if not any(max(abs(pos[0] - q[0]), abs(pos[1] - q[1])) <= 2 * rem for q in cands):
                return

        def try_move(dst, mtype, s2):
            g = region[dst]
            newtower = (mtype == "up") or (mtype == "level" and h == 2)
            if newtower:
                if g in towers:
                    return
                towers[g] = dst
                h2 = 2
            else:
                h2 = 1
            if is_cp:
                ok = dst in NUM and NUM[dst] == s2 and NUM[dst] not in used
            else:
                ok = dst not in NUM
            if ok:
                visited.add(dst)
                heights[dst] = h2
                dead = g not in towers and all(p in visited for p in cells_of[g])
                if not dead:
                    if is_cp:
                        used.add(NUM[dst])
                    path.append(dst)
                    scores.append(s2)
                    if len(towers) == 13 or feas(N, s2, h2, frozenset(used)):
                        dfs(N, dst, s2)
                    path.pop()
                    scores.pop()
                    if is_cp:
                        used.discard(NUM[dst])
                visited.discard(dst)
                heights.pop(dst)
            if newtower:
                del towers[g]

        for d in KNIGHT:
            dst = (pos[0] + d[0], pos[1] + d[1])
            if inb(dst) and dst not in visited:
                try_move(dst, "level", score + N)
        if h == 1 and score * N <= CAP:
            for d in UPDOWN:
                dst = (pos[0] + d[0], pos[1] + d[1])
                if inb(dst) and dst not in visited:
                    try_move(dst, "up", score * N)
        if h == 2 and score % N == 0:
            for d in UPDOWN:
                dst = (pos[0] + d[0], pos[1] + d[1])
                if inb(dst) and dst not in visited:
                    try_move(dst, "down", score // N)

    for h0 in (1, 2):
        heights.clear()
        towers.clear()
        used.clear()
        visited.clear()
        visited.add(START)
        heights[START] = h0
        if h0 == 2:
            towers[region[START]] = START
        path[:] = [START]
        scores[:] = [0]
        dfs(0, START, 0)


print("\n=== Board search ===")
# Score-space already tells us which K can even exist; still search every
# surviving K so the uniqueness claim is on the actual knight path.
search_Ks = sorted({K for K, _ in forced_K_order}) or list(range(4, 10))
for K in search_Ks:
    n_before = len(solutions)
    solve_for_K(K)
    print(f"  K = {K}: {len(solutions) - n_before} path(s)")

assert solutions, "no path found"
assert len(solutions) == 1, f"expected a unique path, got {len(solutions)}"
K, path, scores, towers = solutions[0]
print(f"\nUnique path: K = {K}, {len(path) - 1} moves, ends at {path[-1]}")


# ---------------------------------------------------------------------------
# 4. Independent verification against the raw rules
# ---------------------------------------------------------------------------
tower_set = set(towers.values())
hgt = lambda p: 2 if p in tower_set else 1

assert len(path) == len(set(path)), "revisited a square"
s = 0
recomputed = [0]
for i in range(1, len(path)):
    a, b = path[i - 1], path[i]
    d = sorted([abs(a[0] - b[0]), abs(a[1] - b[1]), abs(hgt(a) - hgt(b))])
    assert d == [0, 1, 2], f"illegal 3D-knight move {i}: {a} -> {b} (delta {d})"
    if hgt(a) == hgt(b):
        s += i
    elif hgt(b) > hgt(a):
        s *= i
    else:
        assert s % i == 0, f"illegal down-move at {i}"
        s //= i
    recomputed.append(s)
assert recomputed == scores, "score recurrence mismatch"

cps = [3, 6, 9, 12, 15, 18] + [18 + i * K for i in range(1, 6)]
for t in cps:
    assert t < len(path)
    assert path[t] in NUM and NUM[path[t]] == scores[t]
for i, p in enumerate(path):
    if p in NUM:
        assert i in cps or i == 0, "numbered square visited off-checkpoint"

regs: dict[str, list] = {}
for p in tower_set:
    regs.setdefault(region[p], []).append(p)
assert sorted(regs) == sorted(cells_of)
assert all(len(v) == 1 for v in regs.values())
assert tower_set <= set(path)
assert path[-1] in tower_set
print("All rule checks passed.")
print("Towers (region -> square):", dict(sorted(towers.items())))

print("\nMove-by-move:")
cpset = set(cps)
for i, (p, sc) in enumerate(zip(path, scores)):
    tag = "  <- checkpoint" if i in cpset else ("  (start)" if i == 0 else "")
    tw = "  [tower]" if p in tower_set else ""
    print(f"  move {i:2d}: {p}  score {sc}{tw}{tag}")


# ---------------------------------------------------------------------------
# 5. Answer: neighbour sums of unvisited squares
# ---------------------------------------------------------------------------
sc = {p: s for p, s in zip(path, scores)}
visited = set(path)
total = 0
print("\nUnvisited squares and their orthogonal neighbour sums:")
for r in range(8):
    for c in range(8):
        if (r, c) not in visited:
            ns = sum(
                sc[q]
                for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if q in sc
            )
            total += ns
            print(f"  {(r, c)} -> {ns}")

print(f"\nANSWER: {total}")
