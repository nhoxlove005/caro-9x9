import numpy as np

class Gomoku:
    def __init__(self) :
        self.row_count = 9
        self.column_count = 9
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "Gomoku"

    def get_initial_state(self):
        # return initial board (numpy board)
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        # return board after applying action
        row = action // self.column_count
        col = action % self.column_count
        state[row][col] = player
        return state
    
    def get_valid_moves(self, state):
        # return valid moves
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_winner(self, state, action):
        if action == None:
            return False
        # return True if player wins
        row = action // self.column_count
        col = action % self.column_count
        player = state[row][col]

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for dir in [-1, 1]:
                r, c = row + dir * dr, col + dir * dc
                while (0 <= r < self.row_count and 0 <= c < self.column_count and state[r][c] == player):
                    count += 1
                    r, c = r + dir * dr, c + dir * dc
            if count >= 5:
                return True

        return False
    
    def get_value_and_terminated(self, state, action):
        # return value and terminal
        if self.check_winner(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        # return opponent
        return -player
    
    def get_opponent_value(self, value):
        # return opponent value
        return -value
    
    def change_perspective(self, state, player):
        # return state after changing perspective
        return state * player
    
    def get_encoded_state(self, state):
        # return encoded state
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
