import pickle
import time
from QNetWork import *

EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.001

# Create an instance of the QNetwork class
q_network = QNetwork()
q_network.to(device)
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
    seed = str(time.time())
    cards = Board.get_card(seed)
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
    if board.get_score() > 100 or (episode+1) % 100 == 0:
        print(f"Episode {episode+1}: Score {board.get_score()} seed {seed}")

    if (episode+1) % 100 == 0:
        with open('q_network_params.pkl', 'wb') as f:
            pickle.dump(q_network.state_dict(), f)

with open('q_network_params.pkl', 'wb') as f:
    pickle.dump(q_network.state_dict(), f)
