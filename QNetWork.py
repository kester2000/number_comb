from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from board import *


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(83, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


def epsilon_greedy(q_values, actions, epsilon):
    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        values = q_values
        max_value = max(values)
        count = values.count(max_value)
        if count > 1:
            best_actions = [action for value, action in zip(
                values, actions) if value == max_value]
            action = random.choice(best_actions)
        else:
            action = actions[values.index(max_value)]
    return action


def get_tensor(board, card, action):
    tensor = []

    for i in board.bin.num:
        tensor.append(i)

    # 19 card
    for i in range(1, 20):
        x, y = move_to_location(i)
        for j in range(3):
            tensor.append(board.matrix[x][y].num[j])

    for i in card.num:
        tensor.append(i)

    tensor.extend([1 if i == action else 0 for i in range(20)])

    return torch.tensor(tensor).float().to(device)


def get_reward(board: Board, card: Card, action):
    board2 = deepcopy(board)
    board2.do_move(card, action)
    return (board2.get_score() - board2.get_potential() / 3) - (board.get_score() - board.get_potential() / 3)
