from time import time
from math import log, sqrt
from copy import deepcopy
from board import *

C_PUCT = 2


def get_random_card(board: Board):
    cards = []
    if not board.bin.is_empty():
        cards.append(board.bin)
    for i in range(1, 20):
        x, y = move_to_location(i)
        if not board.matrix[x][y].is_empty():
            cards.append(board.matrix[x][y])
    cnt = len(cards)
    wild_cnt = sum(1 if card.is_wild() else 0 for card in cards)
    if wild_cnt == 0:
        if random.random() * (cnt - 91 * cnt + 1540) < (90 - 2 * cnt):
            return Card(-1, -1, -1)
    elif wild_cnt == 0:
        if random.random() * (46 - cnt) > 1:
            return Card(-1, -1, -1)
    rest_cards = all_cards[:-1] * 2
    for card in cards:
        if not card.is_wild():
            rest_cards.remove(card)
    return random.choice(rest_cards)


def get_action_by_potential(board: Board, card: Card):
    actions = board.get_actions()
    scores = []
    for action in actions:
        board0 = deepcopy(board)
        board0.do_move(card, action)
        scores.append(board0.get_potential())
    max_value = max(scores)
    return actions[scores.index(max_value)]


def get_mcts_action(board: Board, card: Card, cal_time=5):
    actions_mp = {action: [0, 0] for action in board.get_actions()}
    time0 = time()
    while time()-time0 < cal_time:
        log_sum = sum(actions_mp[a][1] for a in board.get_actions())
        log_sum = log(log_sum) + 1 if log_sum > 0 else 1
        action = random.choices(board.get_actions(), [(actions_mp[a][0] / actions_mp[a][1] if actions_mp[a]
                                                      [1] > 0 else 0) + C_PUCT * sqrt(log_sum / (actions_mp[a][1] + 1)) for a in board.get_actions()])[0]
        board0 = deepcopy(board)
        board0.do_move(card, action)
        while board0.get_actions():
            next_card = get_random_card(board0)
            next_action = get_action_by_potential(board0, next_card)
            board0.do_move(next_card, next_action)
        actions_mp[action][0] += board0.get_score()
        actions_mp[action][1] += 1
    log_sum = sum(actions_mp[a][1] for a in board.get_actions())
    log_sum = log(log_sum) + 1 if log_sum > 0 else 1
    action = random.choices(board.get_actions(), [(actions_mp[a][0] / actions_mp[a][1] if actions_mp[a]
                            [1] > 0 else 0) + C_PUCT * sqrt(log_sum / (actions_mp[a][1] + 1)) for a in board.get_actions()])[0]
    return action
