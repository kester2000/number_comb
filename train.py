import pickle
import time
from QNetWork import *

EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.001

# Create an instance of the QNetwork class
q_network = QNetwork()
q_network.to(device)
episode = 0
try:
    with open('q_network_params.pkl', 'rb') as f:
        saved_params, episode = pickle.load(f)
        q_network.load_state_dict(saved_params)
        print(f'coninue in episode {episode}')
except FileNotFoundError:
    pass

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
seed = 'unknown'

# Run the SARSA algorithm
while True:
    episode += 1
    # Initialize the state and action
    board = Board()
    # seed = str(time.time())
    # cards = Board.get_card(seed)
    cards = Board.get_random_cards()
    state = get_state(board, cards[0])
    q_values = q_network(state)
    action = epsilon_greedy(q_values, board.get_actions(), EPSILON)
    # Repeat until the end of the game
    while True:
        # Execute the action and observe the next state and reward
        board.do_move(cards[0], action)
        cards.pop(0)
        reward = get_reward(board)
        q_values = q_network(state)
        q_value = q_values[action]
        if len(cards) > 0:
            next_state = get_state(board, cards[0])
            next_q_values = q_network(next_state)
            next_action = epsilon_greedy(
                next_q_values, board.get_actions(), EPSILON)
            next_q_value = next_q_values[next_action]
            target_q_value = reward + GAMMA * next_q_value
        else:
            target_q_value = torch.tensor(reward).float().to(device)
        loss = criterion(q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Transition to the next state and action
        state = next_state
        action = next_action
        if len(cards) == 0:
            break

    if episode % 100 == 0:
        with open('q_network_params.pkl', 'wb') as f:
            pickle.dump((q_network.state_dict(), episode), f)

    # Print the total score for the episode
    if board.get_score() > 100 or episode % 100 == 0:
        print(f"Episode {episode}: Score {board.get_score()} seed {seed}")
