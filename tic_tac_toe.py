import numpy as np
import pickle
import time
import threading
from typing import Tuple

# Define the State class (same as your version)
class State:
    def __init__(self, board_size: int, win_length: int):
        self.board_size = board_size
        self.win_length = win_length
        self.data = np.zeros((board_size, board_size))
        self.winner = None
        self.hash_val = None
        self.end = None
    
    def hash(self) -> int:
        if self.hash_val is None:
            self.hash_val = hash(self.data.tobytes())
        return self.hash_val
    
    def is_end(self) -> bool:
        if self.end is not None:
            return self.end
        
        for i in range(self.board_size):
            if self.check_line(self.data[i, :]) or self.check_line(self.data[:, i]):
                self.end = True
                return True
        
        diags = [self.data.diagonal(), np.fliplr(self.data).diagonal()]
        for diag in diags:
            if self.check_line(diag):
                self.end = True
                return True
        
        if np.count_nonzero(self.data) == self.board_size ** 2:
            self.winner = 0
            self.end = True
            return True
        
        self.end = False
        return False
    
    def check_line(self, line: np.ndarray) -> bool:
        for i in range(len(line) - self.win_length + 1):
            if abs(sum(line[i:i + self.win_length])) == self.win_length:
                self.winner = 1 if sum(line[i:i + self.win_length]) > 0 else -1
                return True
        return False
    
    def next_state(self, i: int, j: int, symbol: int) -> 'State':
        new_state = State(self.board_size, self.win_length)
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

# Define the Player class
class Player:
    def __init__(self, board_size: int, win_length: int, step_size=0.1, epsilon=0.1):
        self.board_size = board_size
        self.win_length = win_length
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def set_state(self, state: State):
        self.states.append(state)
        self.greedy.append(True)
    
    def act(self) -> Tuple[int, int, int]:
        state = self.states[-1]
        next_positions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if state.data[i, j] == 0]
        next_states = [state.next_state(i, j, self.symbol).hash() for i, j in next_positions]
        
        for hash_val in next_states:
            if hash_val not in self.estimations:
                self.estimations[hash_val] = 0.5
        
        if np.random.rand() < self.epsilon:
            action = list(next_positions[np.random.randint(len(next_positions))])
            action.append(self.symbol)
            self.greedy[-1] = False
            return tuple(action)
        
        values = [(self.estimations[hash_val], pos) for hash_val, pos in zip(next_states, next_positions)]
        values.sort(reverse=True, key=lambda x: x[0])
        action = list(values[0][1])
        action.append(self.symbol)
        return tuple(action)

# Define the Judger class
class Judger:
    def __init__(self, player1: Player, player2: Player, board_size: int, win_length: int):
        self.p1 = player1
        self.p2 = player2
        self.board_size = board_size
        self.win_length = win_length
        self.p1.set_symbol(1)
        self.p2.set_symbol(-1)
        self.current_state = State(board_size, win_length)
    
    def play(self) -> int:
        self.p1.states.clear()
        self.p2.states.clear()
        current_state = State(self.board_size, self.win_length)
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        
        while True:
            for player in [self.p1, self.p2]:
                i, j, symbol = player.act()
                next_state = current_state.next_state(i, j, symbol)
                current_state = next_state
                self.p1.set_state(current_state)
                self.p2.set_state(current_state)
                if current_state.is_end():
                    return current_state.winner

# Train function with multithreading

def train_worker(board_size: int, win_length: int, epochs: int, thread_id: int):
    player1 = Player(board_size, win_length, epsilon=0.05)
    player2 = Player(board_size, win_length, epsilon=0.05)
    judger = Judger(player1, player2, board_size, win_length)
    
    for _ in range(epochs):
        _ = judger.play()


def train(board_size: int, win_length: int, epochs: int, num_threads: int = 4):
    start_time = time.time()
    threads = []
    epochs_per_thread = epochs // num_threads
    
    for i in range(num_threads):
        thread = threading.Thread(target=train_worker, args=(board_size, win_length, epochs_per_thread, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
