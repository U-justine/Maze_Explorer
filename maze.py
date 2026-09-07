"""
maze.py
-------
Maze generation (recursive-backtracker / DFS) and solving (BFS) utilities.

A maze of size N x N cells is generated. Internally we track connectivity
as a graph: each cell (r, c) knows which of its neighbours (up/down/left/right)
it is connected to (i.e. no wall between them). This makes both rendering
and pathfinding simple.
"""

import random
from collections import deque

# Directions: name -> (delta_row, delta_col)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class Maze:
    """A perfect maze (exactly one path between any two cells)."""

    def __init__(self, size: int = 10, seed: int | None = None):
        self.size = size
        self.rng = random.Random(seed)
        # connections[r][c] is a set of directions in which the cell is OPEN
        # (i.e. you can walk through, no wall).
        self.connections = [[set() for _ in range(size)] for _ in range(size)]
        self.start = (0, 0)
        self.end = (size - 1, size - 1)
        self._generate()
        self.solution_path = self.solve(self.start, self.end)
        self.optimal_length = len(self.solution_path) - 1  # number of moves

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def _generate(self):
        """Recursive-backtracker (iterative, stack-based) maze carving."""
        visited = [[False] * self.size for _ in range(self.size)]
        stack = [self.start]
        visited[self.start[0]][self.start[1]] = True

        while stack:
            r, c = stack[-1]
            neighbours = []
            for direction, (dr, dc) in DIRECTIONS.items():
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and not visited[nr][nc]:
                    neighbours.append((direction, nr, nc))

            if not neighbours:
                stack.pop()
                continue

            direction, nr, nc = self.rng.choice(neighbours)
            # carve a passage both ways
            self.connections[r][c].add(direction)
            self.connections[nr][nc].add(OPPOSITE[direction])
            visited[nr][nc] = True
            stack.append((nr, nc))

    # ------------------------------------------------------------------ #
    # Movement helpers
    # ------------------------------------------------------------------ #
    def can_move(self, pos, direction):
        r, c = pos
        return direction in self.connections[r][c]

    def move(self, pos, direction):
        """Return the new position if the move is legal, else None."""
        if not self.can_move(pos, direction):
            return None
        dr, dc = DIRECTIONS[direction]
        return (pos[0] + dr, pos[1] + dc)

    def neighbours_of(self, pos):
        r, c = pos
        return [self.move(pos, d) for d in self.connections[r][c]]

    def degree(self, pos):
        """Number of open passages from this cell (used to detect dead ends)."""
        r, c = pos
        return len(self.connections[r][c])

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #
    def solve(self, start, end):
        """BFS shortest path between two cells. Returns list of cells."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for direction in self.connections[r][c]:
                nxt = self.move(cur, direction)
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    if nxt == end:
                        # reconstruct
                        path = [nxt]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    queue.append(nxt)
        return []  # unreachable (shouldn't happen in a perfect maze)

    def next_hint_step(self, current_pos):
        """Direction of the next step from current_pos towards the end."""
        path = self.solve(current_pos, self.end)
        if len(path) < 2:
            return None, None
        nxt = path[1]
        dr, dc = nxt[0] - current_pos[0], nxt[1] - current_pos[1]
        for name, (ddr, ddc) in DIRECTIONS.items():
            if (ddr, ddc) == (dr, dc):
                return name, nxt
        return None, None