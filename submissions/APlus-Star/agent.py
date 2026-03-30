import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# Thêm src vào đường dẫn để import được framework
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent: Sử dụng thuật toán Greedy Best-First Search (GBFS) để tìm đường đi đến vị trí tĩnh của Ghost.
    Nếu bị mất dấu (khi có fog of war), nó sẽ đi về vị trí nhìn thấy cuối cùng.
    Khi hoàn toàn mất dấu thì sẽ explore (đi ngẫu nhiên có chủ ý).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Baseline GBFS Pacman"
        self.last_known_enemy_pos = None
        self.last_move = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # Cập nhật vị trí đối thủ nếu nhìn thấy
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        # Đã tới chỗ cuối cùng thấy target nhưng không thấy ai -> xóa target
        if target == my_position:
            self.last_known_enemy_pos = None
            target = None

        if target is not None:
            # Tìm đường GBFS tới target
            path = self._gbfs(my_position, target, map_state)
            if path:
                first_move = path[0]
                self.last_move = first_move
                # Tận dụng tốc độ bằng cách tính số bước có thể tiến thẳng an toàn
                steps = self._max_valid_steps(my_position, first_move, map_state, self.pacman_speed)
                return (first_move, steps)
        
        # Explore ngẫu nhiên nếu không có target
        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (my_position[0] + m.value[0], my_position[1] + m.value[1])
            if self._is_valid_position(nxt, map_state):
                valid_moves.append(m)
        
        if valid_moves:
            # Hạn chế quay đầu khi explore nếu có thể
            if self.last_move is not None and len(valid_moves) > 1:
                opposite_move = self._get_opposite_move(self.last_move)
                if opposite_move in valid_moves:
                    valid_moves.remove(opposite_move)
                    
            move = random.choice(valid_moves)
            self.last_move = move
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            return (move, steps)
        
        self.last_move = Move.STAY
        return (Move.STAY, 1)

    def _get_opposite_move(self, move):
        if move == Move.UP: return Move.DOWN
        if move == Move.DOWN: return Move.UP
        if move == Move.LEFT: return Move.RIGHT
        if move == Move.RIGHT: return Move.LEFT
        return Move.STAY

    def _gbfs(self, start, goal, map_state):
        import heapq
        
        def heuristic(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        queue = []
        # (heuristic, current_pos, path)
        heapq.heappush(queue, (heuristic(start, goal), start, []))
        visited = {start}
        
        while queue:
            _, current, path = heapq.heappop(queue)
            if current == goal:
                return path
            
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (current[0] + m.value[0], current[1] + m.value[1])
                if nxt not in visited and self._is_valid_position(nxt, map_state):
                    # Phạt việc quay ngược hướng ngay bước đầu tiên để tránh oscillation
                    penalty = 0
                    if len(path) == 0 and self.last_move == self._get_opposite_move(m):
                        penalty = 5  # Phạt nặng quay đầu để agent cố gắng tiến lên hoặc rẽ
                        
                    visited.add(nxt)
                    heapq.heappush(queue, (heuristic(nxt, goal) + penalty, nxt, path + [m]))
        return []

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _max_valid_steps(self, pos, move, map_state, max_steps):
        steps = 0
        curr = pos
        for _ in range(max_steps):
            nxt = (curr[0] + move.value[0], curr[1] + move.value[1])
            if not self._is_valid_position(nxt, map_state):
                break
            steps += 1
            curr = nxt
        return max(1, steps)


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent: Trốn tránh Pacman bằng thuật toán Minimax (độ sâu cố định).
    Hàm đánh giá sử dụng khoảng cách Manhattan giữa khoảng cách của Ghost và Pacman.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax Ghost"
        self.last_known_enemy_pos = None
        self.last_move = None

    def _get_opposite_move(self, move):
        if move == Move.UP: return Move.DOWN
        if move == Move.DOWN: return Move.UP
        if move == Move.LEFT: return Move.RIGHT
        if move == Move.RIGHT: return Move.LEFT
        return Move.STAY
        
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos

        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (my_position[0] + m.value[0], my_position[1] + m.value[1])
            if self._is_valid_position(nxt, map_state):
                valid_moves.append(m)

        if not valid_moves:
            self.last_move = Move.STAY
            return Move.STAY # Kẹt cứng

        # Nếu không có threat nào trong lịch sử thì đi tự do
        if threat is None:
            # Hạn chế quay đầu
            if self.last_move is not None and len(valid_moves) > 1:
                opposite_move = self._get_opposite_move(self.last_move)
                if opposite_move in valid_moves:
                    valid_moves.remove(opposite_move)
            move = random.choice(valid_moves)
            self.last_move = move
            return move

        # Tính bản đồ khoảng cách đường đi thực tế thay vì dùng khoảng cách Manhattan
        pacman_distances = self._get_distances_from(threat, map_state)

        # Cài đặt Minimax với depth = 2 (1 lượt Ghost - Max, 1 lượt Pacman - Min)
        best_move = None
        best_score = -float('inf')

        for move in valid_moves:
            nxt_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            # Thêm một chút random nhòe (noise) nhỏ để phá vỡ các trường hợp hòa điểm (nguyên nhân gây đi qua lại)
            score = self._minimax(nxt_pos, threat, map_state, depth=2, is_maximizing=False, pacman_distances=pacman_distances)
            score += random.uniform(0, 0.1)
            
            # Phạt quay đầu để phá vỡ vòng lặp (oscillation)
            if self.last_move == self._get_opposite_move(move):
                score -= 3.0
            
            if score > best_score:
                best_score = score
                best_move = move
                
        if best_move:
            self.last_move = best_move
            return best_move
            
        move = random.choice(valid_moves)
        self.last_move = move
        return move

    def _minimax(self, ghost_pos, pacman_pos, map_state, depth, is_maximizing, pacman_distances):
        # Hệ số phạt nếu bị bắt
        if ghost_pos == pacman_pos:
            return -9999
            
        if depth == 0:
            # Dùng khoảng cách thực tế (BFS) từ Pacman thay cho Manhattan
            return pacman_distances.get(ghost_pos, 9999) + (abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])) * 0.1
            
        if is_maximizing:
            # Lượt của Ghost (muốn khoảng cách xa nhất)
            max_eval = -float('inf')
            
            valid_moves = False
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (ghost_pos[0] + m.value[0], ghost_pos[1] + m.value[1])
                if self._is_valid_position(nxt, map_state):
                    valid_moves = True
                    eval_score = self._minimax(nxt, pacman_pos, map_state, depth - 1, False, pacman_distances)
                    max_eval = max(max_eval, eval_score)
                    
            if not valid_moves: # Bị kẹt
                return self._minimax(ghost_pos, pacman_pos, map_state, depth - 1, False, pacman_distances)
                
            return max_eval
        else:
            # Lượt của Pacman (muốn rút ngắn khoảng cách nhất)
            min_eval = float('inf')
            
            valid_moves = False
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (pacman_pos[0] + m.value[0], pacman_pos[1] + m.value[1])
                if self._is_valid_position(nxt, map_state):
                    valid_moves = True
                    eval_score = self._minimax(ghost_pos, nxt, map_state, depth - 1, True, pacman_distances)
                    min_eval = min(min_eval, eval_score)
                    
            if not valid_moves: # Bị kẹt
                return self._minimax(ghost_pos, pacman_pos, map_state, depth - 1, True, pacman_distances)
                
            return min_eval

    def _get_distances_from(self, start, map_state):
        from collections import deque
        queue = deque([(start, 0)])
        distances = {start: 0}
        
        while queue:
            current, dist = queue.popleft()
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (current[0] + m.value[0], current[1] + m.value[1])
                if nxt not in distances and self._is_valid_position(nxt, map_state):
                    distances[nxt] = dist + 1
                    queue.append((nxt, dist + 1))
        return distances

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0
