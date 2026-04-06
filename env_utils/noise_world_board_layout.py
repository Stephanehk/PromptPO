"""
Numpy-only NoiseWorld layout sampling and per-cell ``board`` coding.

Shared by ``NoiseWorldEnv`` so board matrices can be generated without importing Gymnasium.

Assumptions:
- Same sampling rules and codes as ``noise_world_env.NoiseWorldEnv``.
"""

from collections import deque

import numpy as np

CELL_TYPE_DETERMINISTIC = 1
CELL_TYPE_STOCH_STAY = 2
CELL_TYPE_STOCH_RESET = 3

BOARD_DETERMINISTIC = 0
BOARD_STOCH_STAY = 1
BOARD_STOCH_RESET = 2
BOARD_WALL = 3
BOARD_GOAL = 4
BOARD_BAD_TERMINAL = 5
# Prerequisite milestones (boards 4–6): visit cell with BOARD_PREREQ_FIRST, then
# BOARD_PREREQ_SECOND, before the goal grants its usual positive bonus.
BOARD_PREREQ_FIRST = 6
BOARD_PREREQ_SECOND = 7
BOARD_MAX_CODE = BOARD_PREREQ_SECOND


def _path_exists(n, start, goal, walls):
    if start == goal:
        return True
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            if (nr, nc) in walls:
                continue
            if (nr, nc) in seen:
                continue
            if (nr, nc) == goal:
                return True
            seen.add((nr, nc))
            stack.append((nr, nc))
    return False


def _voronoi_regions(n, rng):
    num_regions = max(4, n // 2)
    seeds = []
    for _ in range(num_regions):
        seeds.append((int(rng.integers(0, n)), int(rng.integers(0, n))))
    region_id = np.zeros((n, n), dtype=np.int32)
    for r in range(n):
        for c in range(n):
            best = 0
            best_d = n + n
            for k, (sr, sc) in enumerate(seeds):
                d = abs(r - sr) + abs(c - sc)
                if d < best_d:
                    best_d = d
                    best = k
            region_id[r, c] = best
    return region_id


def _sample_region_params(num_regions, rng):
    cell_type = np.zeros(num_regions, dtype=np.int32)
    p_succ = np.zeros(num_regions, dtype=np.float64)
    p_alt = np.zeros(num_regions, dtype=np.float64)
    for i in range(num_regions):
        cell_type[i] = int(rng.integers(1, 4))
        p_succ[i] = float(rng.uniform(0.25, 0.95))
        p_alt[i] = float(rng.uniform(0.25, 0.95))
    return cell_type, p_succ, p_alt


def _build_sparse_walls(n, start, goal, rng, max_wall_cells):
    candidates = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) != start and (r, c) != goal
    ]
    rng.shuffle(candidates)
    walls = set()
    for cell in candidates:
        if len(walls) >= max_wall_cells:
            break
        walls.add(cell)
        if not _path_exists(n, start, goal, walls):
            walls.remove(cell)
    return walls


def _sample_negative_terminals(n, walls, start, goal, rng, num_bad):
    free = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) not in walls and (r, c) != start and (r, c) != goal
    ]
    rng.shuffle(free)
    assert len(free) >= num_bad, (
        "Not enough free cells for negative terminals; reduce num_bad or walls."
    )
    return set(free[:num_bad])


def _cell_params_at(region_id, cell_type, r, c):
    rid = int(region_id[r, c])
    return int(cell_type[rid]), float("nan"), float("nan")


def build_board_flat(n, goal, walls, bad_cells, region_id, cell_type):
    """
    Per-cell integer codes for the full grid, row-major length n*n.

    Assumptions:
    - wall / goal / badCells precedence matches ``NoiseWorldEnv``.
    """
    board = np.zeros(n * n, dtype=np.float32)
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            if (r, c) in walls:
                code = BOARD_WALL
            elif (r, c) == goal:
                code = BOARD_GOAL
            elif (r, c) in bad_cells:
                code = BOARD_BAD_TERMINAL
            else:
                ctype, _, _ = _cell_params_at(region_id, cell_type, r, c)
                if ctype == CELL_TYPE_DETERMINISTIC:
                    code = BOARD_DETERMINISTIC
                elif ctype == CELL_TYPE_STOCH_STAY:
                    code = BOARD_STOCH_STAY
                elif ctype == CELL_TYPE_STOCH_RESET:
                    code = BOARD_STOCH_RESET
                else:
                    assert False, "unknown CELL_TYPE_*"
            board[idx] = float(code)
    return board


def build_initial_layout(n, rng):
    """
    Sample regions, walls, bad cells, and board flat vector.

    Returns dict with keys: region_id, cell_type, p_succ, p_alt, walls, bad_cells, board_flat.
    """
    start = (0, 0)
    goal = (n - 1, n - 1)
    if n == 1:
        region_id = np.zeros((1, 1), dtype=np.int32)
        cell_type = np.array([CELL_TYPE_DETERMINISTIC], dtype=np.int32)
        p_succ = np.array([1.0], dtype=np.float64)
        p_alt = np.array([1.0], dtype=np.float64)
        walls = set()
        bad_cells = set()
        assert _path_exists(n, start, goal, walls)
        board_flat = build_board_flat(n, goal, walls, bad_cells, region_id, cell_type)
        return {
            "region_id": region_id,
            "cell_type": cell_type,
            "p_succ": p_succ,
            "p_alt": p_alt,
            "walls": walls,
            "bad_cells": bad_cells,
            "board_flat": board_flat,
        }

    region_id = _voronoi_regions(n, rng)
    num_regions = int(region_id.max()) + 1
    cell_type, p_succ, p_alt = _sample_region_params(num_regions, rng)
    max_walls = max(1, int(0.005 * n * n))
    walls = _build_sparse_walls(n, start, goal, rng, max_walls)
    num_bad = max(1, int(0.002 * n * n))
    bad_cells = _sample_negative_terminals(n, walls, start, goal, rng, num_bad)
    assert _path_exists(n, start, goal, walls), "Invariant: path start→goal."
    board_flat = build_board_flat(n, goal, walls, bad_cells, region_id, cell_type)
    return {
        "region_id": region_id,
        "cell_type": cell_type,
        "p_succ": p_succ,
        "p_alt": p_alt,
        "walls": walls,
        "bad_cells": bad_cells,
        "board_flat": board_flat,
    }


def _cells_reachable_from_start(n, start, walls):
    """
    All cells reachable from ``start`` by 4-neighbor moves through non-wall cells.

    Assumptions:
    - start is inside the grid; walls block movement only.
    """
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            if (nr, nc) in walls:
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def inject_prerequisite_pair(n, board_flat, walls, bad_cells, goal, start, rng):
    """
    Place BOARD_PREREQ_FIRST and BOARD_PREREQ_SECOND on two distinct reachable free cells.

    Each code appears exactly once. Cells may not be start, goal, wall, or bad terminal.
    Dynamics on these cells are handled in the env (deterministic moves).

    Inputs:
    - n: grid side length.
    - board_flat: length n*n layout vector (copied, not mutated in place).
    - walls, bad_cells, goal, start: same meaning as ``build_initial_layout``.
    - rng: numpy Generator.

    Outputs:
    - new_board_flat: updated flat board.
    - pos_first: (r, c) where code BOARD_PREREQ_FIRST is placed (visit before second).
    - pos_second: (r, c) for BOARD_PREREQ_SECOND.

    Assumptions:
    - At least two valid placement cells exist (n >= 2 typical for n=10 boards).
    """
    reachable = _cells_reachable_from_start(n, start, walls)
    candidates = []
    for r in range(n):
        for c in range(n):
            p = (r, c)
            if p not in reachable:
                continue
            if p == start or p == goal:
                continue
            if p in bad_cells or p in walls:
                continue
            candidates.append(p)
    assert len(candidates) >= 2, (
        "need at least two free cells for prerequisite milestones (n=%s)" % n
    )
    rng.shuffle(candidates)
    pos_first = candidates[0]
    pos_second = candidates[1]
    out = np.asarray(board_flat, dtype=np.float32).copy()
    i1 = pos_first[0] * n + pos_first[1]
    i2 = pos_second[0] * n + pos_second[1]
    out[i1] = float(BOARD_PREREQ_FIRST)
    out[i2] = float(BOARD_PREREQ_SECOND)
    return out, pos_first, pos_second


def ensure_minimum_standard_board_codes(n, state, rng):
    """
    Mutate ``walls``, ``bad_cells``, and ``cell_type`` so the flattened board has at least one
    cell of each code 0..5 (deterministic / stochastic stay / stochastic reset / wall / goal /
    bad terminal).

    Strategy: sample layout first, then if needed add a wall (preserving a start→goal path),
    add a bad terminal on a free cell, and assign distinct Voronoi regions' ``cell_type`` so
    each of the three free-cell dynamics codes appears on some non-special cell.

    Assumptions:
    - ``state`` is a ``build_initial_layout`` dict; ``n >= 2``.
    - Goal is (n-1, n-1), start (0, 0); matches ``NoiseWorldEnv``.
    """
    goal = (n - 1, n - 1)
    start = (0, 0)
    walls = set(state["walls"])
    bad_cells = set(state["bad_cells"])
    region_id = state["region_id"]
    cell_type = np.array(state["cell_type"], copy=True)
    num_regions = len(cell_type)
    required = frozenset(
        {
            BOARD_DETERMINISTIC,
            BOARD_STOCH_STAY,
            BOARD_STOCH_RESET,
            BOARD_WALL,
            BOARD_GOAL,
            BOARD_BAD_TERMINAL,
        }
    )

    for _ in range(500):
        bf = build_board_flat(n, goal, walls, bad_cells, region_id, cell_type)
        present = set(np.unique(bf.astype(np.int32)))
        if required <= present:
            state["walls"] = walls
            state["bad_cells"] = bad_cells
            state["cell_type"] = cell_type
            state["board_flat"] = bf
            return state
        missing = required - present
        assert BOARD_GOAL in present, "goal cell must always appear in the board layout"

        if BOARD_WALL in missing:
            candidates = [
                (r, c)
                for r in range(n)
                for c in range(n)
                if (r, c) not in walls and (r, c) != start and (r, c) != goal
            ]
            rng.shuffle(candidates)
            placed = False
            for cell in candidates:
                walls.add(cell)
                if _path_exists(n, start, goal, walls):
                    placed = True
                    break
                walls.remove(cell)
            assert placed, "could not add a wall while keeping a path from start to goal"
            continue

        if BOARD_BAD_TERMINAL in missing:
            candidates = [
                (r, c)
                for r in range(n)
                for c in range(n)
                if (r, c) not in walls
                and (r, c) not in bad_cells
                and (r, c) != start
                and (r, c) != goal
            ]
            assert candidates, "no free cell available for a bad terminal"
            bad_cells.add(candidates[int(rng.integers(len(candidates)))])
            continue

        used_rid = set()
        code_to_ctype = {
            BOARD_DETERMINISTIC: CELL_TYPE_DETERMINISTIC,
            BOARD_STOCH_STAY: CELL_TYPE_STOCH_STAY,
            BOARD_STOCH_RESET: CELL_TYPE_STOCH_RESET,
        }
        for board_code in (
            BOARD_DETERMINISTIC,
            BOARD_STOCH_STAY,
            BOARD_STOCH_RESET,
        ):
            if board_code not in missing:
                continue
            ctype = code_to_ctype[board_code]
            rids = [rid for rid in range(num_regions) if rid not in used_rid]
            rng.shuffle(rids)
            placed = False
            for rid in rids:
                for r in range(n):
                    for c in range(n):
                        if int(region_id[r, c]) != rid:
                            continue
                        if (r, c) in walls or (r, c) == goal or (r, c) in bad_cells:
                            continue
                        cell_type[rid] = ctype
                        used_rid.add(rid)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            assert placed, (
                "could not assign a region for board code %s (n=%s, num_regions=%s)"
                % (board_code, n, num_regions)
            )
        continue

    assert False, "ensure_minimum_standard_board_codes did not converge"


def board_matrix_2d(
    n,
    seed,
    ensure_all_standard_board_codes=False,
    prerequisite_pair=False,
):
    """
    Return (n, n) int array of cell codes for NoiseWorld with this ``seed``.

    When ``prerequisite_pair`` is True, two distinct reachable non-special cells are chosen
    at random (using the same RNG stream order as ``NoiseWorldEnv`` after layout sampling)
    and overwritten with ``BOARD_PREREQ_FIRST`` (6) and ``BOARD_PREREQ_SECOND`` (7).

    Assumptions:
    - Matches ``NoiseWorldEnv(n=n, seed=seed, ensure_all_standard_board_codes=...,
      prerequisite_pair=...)`` when the same flags are passed.
    """
    rng = np.random.default_rng(seed)
    state = build_initial_layout(n, rng)
    if ensure_all_standard_board_codes:
        state = ensure_minimum_standard_board_codes(n, state, rng)
    board_flat = state["board_flat"]
    if prerequisite_pair:
        goal = (n - 1, n - 1)
        start = (0, 0)
        board_flat, _, _ = inject_prerequisite_pair(
            n,
            board_flat,
            state["walls"],
            state["bad_cells"],
            goal,
            start,
            rng,
        )
    return board_flat.reshape(n, n).astype(np.int32)
