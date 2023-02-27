import torch
import torch.nn as nn
import numpy as np
from board import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(285, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 20)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
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
    state = []

    # 19 card
    for i in range(1, 20):
        x, y = move_to_location(i)
        for j in range(3):
            state.append(board.matrix[x][y].num[j])

    scores = []
    potentials = []

    # 往右下
    for x, y in [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0]]:
        mx = max(x, y)
        score_list0 = [board.matrix[x + k][y + k].num[0]
                       for k in range(5 - mx)]
        scores.append(get_line_score(score_list0))
        potentials.append(get_line_potential(score_list0))

    # 往下
    for y in range(5):
        score_list1 = [board.matrix[k][y].num[1] for k in range(5)]
        scores.append(get_line_score(score_list1))
        potentials.append(get_line_potential(score_list1))

    # 往右
    for x in range(5):
        score_list2 = [board.matrix[x][k].num[2] for k in range(5)]
        scores.append(get_line_score(score_list2))
        potentials.append(get_line_potential(score_list2))

    for i, score in enumerate(scores):
        state.extend([score] * (5 - abs(i % 5 - 2)))

    for i, potentials in enumerate(potentials):
        state.extend([potentials] * (5 - abs(i % 5 - 2)))

    for i in board.bin.num:
        state.extend([i] * 19)

    for i in card.num:
        state.extend([i] * 19)

    state_tensor = torch.tensor(state).float().to(device)
    return state_tensor


def get_reward(board: Board):
    return board.get_score() + board.get_potential() * 0.5
