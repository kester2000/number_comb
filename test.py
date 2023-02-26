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


card_list = Board.get_card('22073789477')
print(len(card_list))

perms = [
    (0, [4,8,6,10,13,5,16,7,2,12,3,1,9,18,19,14,15,17,0,11]),
]

# print(len(perms))
for perm in perms:
    board = Board()
    for card, move in zip(card_list, perm[1]):
        if move != -1:
            board.do_move(card, move)
    board.show(str(board.get_score()), str(perm[1]))
