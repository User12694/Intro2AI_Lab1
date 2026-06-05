import sys
from pathlib import Path
from collections import deque
import numpy as np
import heapq
import random

# Thêm src vào đường dẫn để import được framework
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Lưu path hiện tại
        self.path = []
        self.path_index = 0

        # Lưu vị trí enemy để detect thay đổi
        self.last_enemy = None

        # Cache ô hợp lệ
        self.valid_cells = set()

    def step(self, map_state, my_pos, enemy_pos, step_number):

        # Precompute 1 lần
        if not self.valid_cells:
            self._build_valid_cells(map_state)

        # Kiểm tra có cần tìm lại đường không
        need_replan = (
            not self.path or
            self.path_index >= len(self.path) or
            enemy_pos != self.last_enemy
        )

        if need_replan:
            self.path = self._astar(my_pos, enemy_pos)
            self.path_index = 0
            self.last_enemy = enemy_pos

        # Nếu có path → đi theo path
        if self.path_index < len(self.path):
            move = self.path[self.path_index]

            steps = self._max_steps(my_pos, move)

            self.path_index += steps
            return (move, steps)

        # fallback: greedy
        move = self._greedy(my_pos, enemy_pos)
        steps = self._max_steps(my_pos, move)

        return (move, max(1, steps))

    # ========================
    # HELPER
    # ========================

    def _build_valid_cells(self, map_state):
        """Lưu tất cả ô đi được"""
        h, w = map_state.shape
        for r in range(h):
            for c in range(w):
                if map_state[r, c] == 0:
                    self.valid_cells.add((r, c))

    def _is_valid(self, pos):
        return pos in self.valid_cells

    def _max_steps(self, pos, move):
        """Tính số bước đi tối đa theo 1 hướng"""
        steps = 0
        cur = pos

        for _ in range(self.speed):
            dr, dc = move.value
            nxt = (cur[0] + dr, cur[1] + dc)

            if not self._is_valid(nxt):
                break

            steps += 1
            cur = nxt

        return steps

    def _astar(self, start, goal):
        """A* dùng heap (chuẩn)"""

        heap = []
        heapq.heappush(heap, (0, start))
        # heapq là thư viện trong Python dùng để làm priority queue (hàng đợi ưu tiên) dựa trên min-heap.
        # Phần tử nhỏ nhất g_cost luôn nằm ở đầu, lấy ra nhanh nhất

        g_cost = {start: 0}
        parent = {}

        while heap:
            _, current = heapq.heappop(heap)

            if current == goal:
                return self._reconstruct(parent, start, goal)

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (current[0] + dr, current[1] + dc)

                if not self._is_valid(nxt):
                    continue

                new_g = g_cost[current] + 1

                if nxt not in g_cost or new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = (current, move)

                    h = self._manhattan(nxt, goal)
                    f = new_g + h

                    heapq.heappush(heap, (f, nxt))

        return []

    def _reconstruct(self, parent, start, goal):
        path = []
        cur = goal

        while cur != start:
            cur, move = parent[cur]
            path.append(move)

        return path[::-1]

    def _greedy(self, my_pos, enemy_pos):
        """Chọn hướng giảm khoảng cách nhanh nhất"""
        best_move = Move.STAY
        best_dist = float('inf')

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_valid(nxt):
                continue

            dist = self._manhattan(nxt, enemy_pos)

            if dist < best_dist:
                best_dist = dist
                best_move = move

        return best_move

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
from collections import deque

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None

    def step(self, map_state, my_pos, enemy_pos, step_number):

        # Tạo distance map từ Pacman
        dist_map = self._bfs(enemy_pos, map_state)

        best_move = Move.STAY
        best_dist = dist_map[my_pos]

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._valid(nxt, map_state):
                continue

            d = dist_map[nxt]

            # Nếu unreachable → coi như rất xa
            if d == -1:
                d = float('inf')

            if d > best_dist:
                best_dist = d
                best_move = move

        return best_move

    # ========================
    # BFS
    # ========================

    def _bfs(self, start, map_state):
        h, w = map_state.shape
        dist = np.full((h, w), -1)

        q = deque()
        # deque (double-ended queue) là hàng đợi thêm/xóa đầu và cuối đều nhanh (hơn list)

        q.append(start)
        dist[start] = 0

        while q:
            cur = q.popleft()

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (cur[0] + dr, cur[1] + dc)

                if self._valid(nxt, map_state) and dist[nxt] == -1:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)

        return dist

    def _valid(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape

        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0