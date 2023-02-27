import torch
import torch.nn as nn
import numpy as np
from board import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(588, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 20)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


def epsilon_greedy(q_values, actions, epsilon):
    # Define the epsilon-greedy policy
    if random.random() < epsilon:
        # Choose a random action
        action = random.choice(actions)
    else:
        # Choose the best action based on the Q-values
        values = [q_values[a] for a in actions]
        max_value = max(values)
        count = values.count(max_value)
        if count > 1:
            # Choose a random action among the best ones
            best_actions = [a for a in actions if q_values[a] == max_value]
            action = random.choice(best_actions)
        else:
            action = actions[values.index(max_value)]
    return action


def get_state(board: Board, card: Card):
    state = np.zeros((28, 21))
    for i in range(28):
        for j in range(21):
            if j == 0:
                if board.bin == all_cards[i]:
                    state[i, j] = 1
            elif 1 <= j <= 19:
                x, y = move_to_location(j)
                if board.matrix[x][y] == all_cards[i]:
                    state[i, j] = 1
            elif j == 20:
                if card == all_cards[i]:
                    state[i, j] = 1
    state_tensor = torch.from_numpy(state.reshape(588)).float().to(device)
    return state_tensor
