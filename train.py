import pickle
import time
from QNetWork import *

EPSILON = 0.1
LEARNING_RATE = 0.001

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
q_network_2 = deepcopy(q_network)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
seed = 'unknown'

batch_buff = []

while True:
    episode += 1
    board = Board()
    # seed = str(time.time())
    # cards = Board.get_card(seed)
    cards = Board.get_random_cards()
    for i in range(20):
        q_values = [q_network_2(get_tensor(board, cards[i], action))
                    for action in board.get_actions()]
        action = epsilon_greedy(q_values, board.get_actions(), EPSILON)
        q_value = q_network(get_tensor(board, cards[i], action))
        reward = get_reward(board, cards[i], action)

        board.do_move(cards[i], action)

        if i < 19:
            next_q_values = [q_network_2(get_tensor(
                board, cards[i + 1], action)) for action in board.get_actions()]
            next_action = epsilon_greedy(
                next_q_values, board.get_actions(), EPSILON)
            next_q_value = q_network_2(get_tensor(board, cards[i+1], action))
            target_q_value = reward + next_q_value
        else:
            target_q_value = torch.tensor([reward]).to(device)

        optimizer.zero_grad()
        loss = criterion(q_value, target_q_value)
        loss.backward()
        optimizer.step()

    if episode % 10 == 0:
        q_network_2 = deepcopy(q_network)

    if episode % 100 == 0:
        with open('q_network_params.pkl', 'wb') as f:
            pickle.dump((q_network.state_dict(), episode), f)

    if board.get_score() > 100 or episode % 100 == 0:
        print(f"Episode {episode}: Score {board.get_score()} seed {seed}")
