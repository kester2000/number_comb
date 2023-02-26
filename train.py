import pickle
import numpy as np
import random
import time
import torch
import torch.nn as nn
from board import *

EPSILON = 0.1
GAMMA = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Define the neural network for function approximation

all_cards = []
for i in [8, 4, 3]:
    for j in [9, 5, 1]:
        for k in [7, 6, 2]:
            all_cards.append(Card(i, j, k))
all_cards.append(Card(-1, -1, -1))


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
    state_tensor = torch.from_numpy(state.reshape(588)).float()
    return state_tensor


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


# Create an instance of the QNetwork class
q_network = QNetwork()
try:
    with open('q_network_params.pkl', 'rb') as f:
        saved_params = pickle.load(f)
        q_network.load_state_dict(saved_params)
except FileNotFoundError:
    pass

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

# Run the SARSA algorithm
for episode in range(1000000):
    # Initialize the state and action
    board = Board()
    cards = Board.get_card(str(time.time()))
    state = get_state(board, cards[0])
    q_values = q_network(state)
    action = epsilon_greedy(q_values, board.get_actions(), EPSILON)
    # Repeat until the end of the game
    while True:
        # Execute the action and observe the next state and reward
        board.do_move(cards[0], action)
        cards.pop(0)
        if len(board.get_actions()) == 0:
            break
        reward = board.get_score()
        q_values = q_network(state)
        q_value = q_values[action]
        next_state = get_state(board, cards[0])
        next_q_values = q_network(next_state)
        next_action = epsilon_greedy(
            next_q_values, board.get_actions(), EPSILON)
        next_q_value = next_q_values[next_action]
        target_q_value = reward + GAMMA * next_q_value
        loss = criterion(q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Transition to the next state and action
        state = next_state
        action = next_action

    # Print the total score for the episode
    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}: Score {board.get_score()}")

    if (episode+1) % 100 == 0:
        with open('q_network_params.pkl', 'wb') as f:
            pickle.dump(q_network.state_dict(), f)

with open('q_network_params.pkl', 'wb') as f:
    pickle.dump(q_network.state_dict(), f)
