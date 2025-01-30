import numpy as np
import pickle
from typing import Tuple, List

class State:
    def __init__(self, board_rows: int, board_cols: int, win_length: int):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.win_length = win_length
        self.data = np.zeros((board_rows, board_cols))
        self.winner = None
        self.hash_val = None
        self.end = None

    def hash(self) -> int:
        if self.hash_val is None:
            # Use a more robust hashing method that's size-independent
            self.hash_val = hash(self.data.tobytes())
        return self.hash_val

    def check_line(self, line: np.ndarray) -> int:
        # Check for win_length consecutive symbols
        count_1 = count_neg_1 = 0
        max_count_1 = max_count_neg_1 = 0
        
        for val in line:
            if val == 1:
                count_1 += 1
                count_neg_1 = 0
            elif val == -1:
                count_neg_1 += 1
                count_1 = 0
            else:
                count_1 = count_neg_1 = 0
            max_count_1 = max(max_count_1, count_1)
            max_count_neg_1 = max(max_count_neg_1, count_neg_1)
            
        if max_count_1 >= self.win_length:
            return 1
        if max_count_neg_1 >= self.win_length:
            return -1
        return 0

    def is_end(self) -> bool:
        if self.end is not None:
            return self.end

        # Check rows
        for i in range(self.board_rows):
            result = self.check_line(self.data[i, :])
            if result != 0:
                self.winner = result
                self.end = True
                return True

        # Check columns
        for i in range(self.board_cols):
            result = self.check_line(self.data[:, i])
            if result != 0:
                self.winner = result
                self.end = True
                return True

        # Check diagonals
        for i in range(self.board_rows - self.win_length + 1):
            for j in range(self.board_cols - self.win_length + 1):
                # Main diagonal
                diag = [self.data[i + k, j + k] for k in range(self.win_length)]
                result = self.check_line(np.array(diag))
                if result != 0:
                    self.winner = result
                    self.end = True
                    return True
                
                # Anti-diagonal
                anti_diag = [self.data[i + k, j + self.win_length - 1 - k] for k in range(self.win_length)]
                result = self.check_line(np.array(anti_diag))
                if result != 0:
                    self.winner = result
                    self.end = True
                    return True

        # Check for tie
        if np.count_nonzero(self.data) == self.board_rows * self.board_cols:
            self.winner = 0
            self.end = True
            return True

        self.end = False
        return False

    def next_state(self, i: int, j: int, symbol: int) -> 'State':
        new_state = State(self.board_rows, self.board_cols, self.win_length)
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def print_state(self):
        for i in range(self.board_rows):
            print("-" * (self.board_cols * 4 + 1))
            out = "| "
            for j in range(self.board_cols):
                if self.data[i, j] == 1:
                    token = "*"
                elif self.data[i, j] == -1:
                    token = "O"
                else:
                    token = " "
                out += token + " | "
            print(out)
        print("-" * (self.board_cols * 4 + 1))

class Player:
    def __init__(self, board_rows: int, board_cols: int, win_length: int, step_size=0.1, epsilon=0.1):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.win_length = win_length
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state: State):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol: int):
        self.symbol = symbol
        
    def backup(self):
        states = [state.hash() for state in self.states]
        for i in reversed(range(len(states) - 1)):
            state = states[i]
            if state not in self.estimations:
                self.estimations[state] = 0.5
            if states[i + 1] not in self.estimations:
                self.estimations[states[i + 1]] = 0.5
            td_error = self.greedy[i] * (self.estimations[states[i + 1]] - self.estimations[state])
            self.estimations[state] += self.step_size * td_error

    def act(self) -> Tuple[int, int, int]:
        state = self.states[-1]
        next_positions = [(i, j) for i in range(self.board_rows) for j in range(self.board_cols) if state.data[i, j] == 0]
        next_states = [state.next_state(i, j, self.symbol).hash() for i, j in next_positions]
        
        # Initialize estimations for new states
        for hash_val in next_states:
            if hash_val not in self.estimations:
                self.estimations[hash_val] = 0.5

        if np.random.rand() < self.epsilon:
            action = list(next_positions[np.random.randint(len(next_positions))])
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = [(self.estimations[hash_val], pos) for hash_val, pos in zip(next_states, next_positions)]
        values.sort(key=lambda x: x[0], reverse=True)
        action = list(values[0][1])
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open(f'policy_{self.board_rows}x{self.board_cols}_{"first" if self.symbol == 1 else "second"}.bin', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        try:
            with open(f'policy_{self.board_rows}x{self.board_cols}_{"first" if self.symbol == 1 else "second"}.bin', 'rb') as f:
                self.estimations = pickle.load(f)
        except FileNotFoundError:
            print(f"No existing policy found for {self.board_rows}x{self.board_cols} board. Starting fresh.")
            self.estimations = dict()

class HumanPlayer:
    def __init__(self, board_rows: int, board_cols: int, win_length: int):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.win_length = win_length
        self.symbol = None
        self.state = None

    def reset(self):
        pass

    def set_state(self, state: State):
        self.state = state

    def set_symbol(self, symbol: int):
        self.symbol = symbol

    def act(self) -> Tuple[int, int, int]:
        while True:
            try:
                self.state.print_state()
                print(f"Enter row (0-{self.board_rows-1}) and column (0-{self.board_cols-1}) separated by space:")
                i, j = map(int, input().split())
                if 0 <= i < self.board_rows and 0 <= j < self.board_cols and self.state.data[i, j] == 0:
                    return i, j, self.symbol
                else:
                    print("Invalid move. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter two numbers separated by space.")

class Judger:
    def __init__(self, player1: Player, player2: Player, board_rows: int, board_cols: int, win_length: int):
        self.p1 = player1
        self.p2 = player2
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.win_length = win_length
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State(board_rows, board_cols, win_length)

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, print_state=False) -> int:
        alternator = self.alternate()
        self.reset()
        current_state = State(self.board_rows, self.board_cols, self.win_length)
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        
        if print_state:
            current_state.print_state()

        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state = current_state.next_state(i, j, symbol)
            current_state = next_state
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            
            if print_state:
                current_state.print_state()
                
            if current_state.is_end():
                return current_state.winner

def train(board_rows: int, board_cols: int, win_length: int, epochs: int, print_every_n=500):
    player1 = Player(board_rows, board_cols, win_length, epsilon=0.01)
    player2 = Player(board_rows, board_cols, win_length, epsilon=0.01)
    judger = Judger(player1, player2, board_rows, board_cols, win_length)
    player1_win = 0.0
    player2_win = 0.0
    
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % 
                  (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()
    
    player1.save_policy()
    player2.save_policy()

def play_human_vs_ai(board_rows: int, board_cols: int, win_length: int):
    while True:
        player1 = HumanPlayer(board_rows, board_cols, win_length)
        player2 = Player(board_rows, board_cols, win_length, epsilon=0)
        player2.load_policy()
        judger = Judger(player1, player2, board_rows, board_cols, win_length)
        
        winner = judger.play(print_state=True)
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It's a tie!")
            
        print("Play again? (y/n)")
        if input().lower() != 'y':
            break

if __name__ == '__main__':
    # Example usage for a 4x4 board with 3 in a row needed to win
    BOARD_ROWS = 4
    BOARD_COLS = 4
    WIN_LENGTH = 3
    
    print(f"Training AI for {BOARD_ROWS}x{BOARD_COLS} board with {WIN_LENGTH} in a row to win...")
    train(BOARD_ROWS, BOARD_COLS, WIN_LENGTH, epochs=100000)
    
    print("\nLet's play!")
    play_human_vs_ai(BOARD_ROWS, BOARD_COLS, WIN_LENGTH)