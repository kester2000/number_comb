from copy import deepcopy
import torch
import itertools
import numpy as np
from board import *

# card_list = [
#     Piece(3, 9, 6),
#     Piece(4, 5, 6),
#     Piece(8, 5, 6),
#     Piece(8, 5, 2),
#     Piece(-1, -1, -1),
#     Piece(-1, -1, -1),
#     Piece(8, 9, 2),
#     Piece(8, 9, 7),
#     Piece(4, 1, 6),
#     Piece(3, 5, 6),
#     Piece(4, 9, 2),
#     Piece(8, 5, 7),
#     Piece(8, 1, 7),
#     Piece(4, 9, 6),
#     Piece(3, 9, 2),
#     Piece(4, 9, 7),
#     Piece(8, 9, 7),
#     Piece(4, 1, 2),
#     Piece(3, 1, 7),
#     Piece(4, 1, 7),
# ]

# card_list = [Piece(-1, -1, -1), Piece(-1, -1, -1)]
# for a in [8, 4, 3]:
#     for b in [9, 5, 1]:
#         for c in [7, 6, 2]:
#             card_list.append(Piece(a, b, c))
#             card_list.append(Piece(a, b, c))


# card_list = Board.get_card('22073789477')
# print(len(card_list))

# perms = [
#     (0, [4,8,6,10,13,5,16,7,2,12,3,1,9,18,19,14,15,17,0,11]),
# ]

# # print(len(perms))
# for perm in perms:
#     board = Board()
#     for card, move in zip(card_list, perm[1]):
#         if move != -1:
#             board.do_move(card, move)
#     board.show(str(board.get_score()), str(perm[1]))

# c1 = Card(-1, -1, -1)
# c2 = Card(-1, -1, -1)
# print(c1 == c2)

# cards = Board.get_random_cards()
# print(len(cards))
# for card in cards:
#     print(card)

# print(sum(card.get_score() for card in all_cards)/len(all_cards))

# board = Board()
# card = Card(8, 9, 7)
# board.do_move(card, 1)
# board.show()
# print(board.get_potential())

id = [list(it) for it in itertools.combinations(range(56), 2)]
l = []
for i in id:
    a = [0]*56
    a[i[0]] = 1
    a[i[1]] = 1
    if i[0] < 20:
        l.append(a[:20])
    else:
        l.append(a[20:40])

for k in range(20):
    cnt = 0
    tot = 0
    for ll in l:
        s = sum(ll[i] for i in range(k))
        if s == 1:
            tot += 1
            if ll[k] == 1:
                cnt += 1
    print(k, cnt, tot)
