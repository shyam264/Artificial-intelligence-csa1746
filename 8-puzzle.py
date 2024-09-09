import heapq

class PuzzleState:
    def __init__(self, board, goal, moves=0, previous=None):
        self.board = board
        self.goal = goal
        self.moves = moves
        self.previous = previous
        self.blank_pos = self.find_blank()
        self.priority = self.moves + self.manhattan()

    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)
        return None

    def manhattan(self):
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    continue
                goal_pos = [(r, c) for r in range(3) for c in range(3) if self.goal[r][c] == self.board[i][j]][0]
                distance += abs(goal_pos[0] - i) + abs(goal_pos[1] - j)
        return distance

    def is_goal(self):
        return self.board == self.goal

    def neighbors(self):
        neighbors = []
        x, y = self.blank_pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(PuzzleState(new_board, self.goal, self.moves + 1, self))
        return neighbors

    def __lt__(self, other):
        return self.priority < other.priority

def solve_8_puzzle(start, goal):
    start_state = PuzzleState(start, goal)
    priority_queue = []
    heapq.heappush(priority_queue, start_state)
    visited = set()

    while priority_queue:
        current_state = heapq.heappop(priority_queue)
        
        if current_state.is_goal():
            return reconstruct_path(current_state)

        visited.add(tuple(map(tuple, current_state.board)))

        for neighbor in current_state.neighbors():
            if tuple(map(tuple, neighbor.board)) not in visited:
                heapq.heappush(priority_queue, neighbor)

    return None

def reconstruct_path(state):
    path = []
    while state:
        path.append(state.board)
        state = state.previous
    return path[::-1]

def print_puzzle(puzzle):
    for row in puzzle:
        print(' '.join(str(n) if n != 0 else ' ' for n in row))
    print()

if __name__ == "__main__":
    start = [
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8]
    ]

    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]

    solution = solve_8_puzzle(start, goal)

    if solution:
        print("Solution found in {} moves!".format(len(solution) - 1))
        for step in solution:
            print_puzzle(step)
    else:
        print("No solution exists.")
