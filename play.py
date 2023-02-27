import pickle
from QNetWork import *

q_network = QNetwork()
q_network.to(device)

with open('q_network_params.pkl', 'rb') as f:
    saved_params, episode = pickle.load(f)
    q_network.load_state_dict(saved_params)

board = Board()
seed = '1677408732.3003042'
cards = Board.get_card(seed)
for i in range(20):
    card = cards[i]
    # x, y, z = map(int, input().split())
    # card = Card(x, y, z)
    state = get_state(board, card)
    q_values = q_network(state)
    action = epsilon_greedy(q_values, board.get_actions(), 0)
    board.do_move(card, action)
    # board.show(str(board.get_score()), str(action))
    # pass
print(board.get_score(), get_reward(board))
