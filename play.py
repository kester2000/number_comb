import pickle
# from QNetWork import *
from mcts import *

# q_network = QNetwork()
# q_network.to(device)

# with open('q_network_params.pkl', 'rb') as f:
#     saved_params, episode = pickle.load(f)
#     q_network.load_state_dict(saved_params)

board = Board()
# seed = '1677408732.3003042'
# cards = Board.get_card(seed)
cards = Board.get_random_cards()
for card in cards:
    print(card)
for i in range(20):

    card = cards[i]
    # x, y, z = map(int, input().split())
    # card = Card(x, y, z)

    # q_values = [q_network(get_tensor(board, card, action))
    #             for action in board.get_actions()]
    # print(q_values)
    # action = epsilon_greedy(q_values, board.get_actions(), 0)
    action = get_mcts_action(board, card, 5)

    board.do_move(card, action)
    print(board.get_score())
    # board.show(str(board.get_score()), str(action))
    # pass
print(board.get_score())
