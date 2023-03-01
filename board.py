import random
import subprocess
from PIL import Image, ImageDraw

MP = [[1, 4, 8, -1, -1], [2, 5, 9, 13, -1], [3, 6, 10, 14, 17],
      [-1, 7, 11, 15, 18], [-1, -1, 12, 16, 19]]
MOVE = [[-1, -1], [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [3, 1], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
        [1, 3], [2, 3], [3, 3], [4, 3], [2, 4], [3, 4], [4, 4]]


class Card:
    # 0,0,0为空 -1,-1,-1为癞子
    def __init__(self, p0=0, p1=0, p2=0) -> None:
        self.num = [p0, p1, p2]

    def is_empty(self) -> bool:
        return all(x == 0 for x in self.num)

    def is_wild(self) -> bool:
        return all(x == -1 for x in self.num)

    # 返回单独分数
    def get_score(self) -> int:
        return 30 if self.is_wild() else sum(self.num)

    def get_image(self, move) -> Image:
        if self.is_wild():
            image = Image.open('resource/card_any.png')
        elif self.is_empty():
            image = Image.open('resource/num_' + str(move) + '.png')
        else:
            image = Image.open(
                'resource/card_' + str(self.num[0]) + str(self.num[1]) + str(self.num[2]) + '.png')
        return image

    def __eq__(self, b):
        return self.num == b.num

    def __str__(self):
        return str(self.num)


def location_to_move(pair: tuple) -> int:
    if len(pair) != 2:
        raise Exception("坐标类型错误")
    if pair == (-1, -1):
        return -1
    r = MP[pair[0]][pair[1]]
    if r == -1:
        raise Exception("坐标位置错误")
    return int(r)


def move_to_location(move: int) -> tuple:
    if not (0 <= move <= 19):
        raise Exception("移动位置错误")
    return MOVE[move]


def get_line_potential(score_list: list) -> int:
    # 得到一行潜力，score_list为该行

    # num为得到第一个非癞子非空
    num = -1
    for i in score_list:
        if i != -1 and i != 0:
            num = i
            break

    if num == -1:
        return 10

    # 转化所有癞子
    change_list = [num if x == -1 else x for x in score_list]

    # 全部相同则得分
    if all(x == num for x in change_list):
        return num

    return 0


def get_line_score(score_list: list) -> int:
    # 得到一行分数，score_list为该行

    # num为得到第一个非癞子
    num = -1
    for i in score_list:
        if i != -1:
            num = i
            break

    if num == -1:
        raise Exception("所有均为癞子")

    # 转化所有癞子
    change_list = [num if x == -1 else x for x in score_list]

    # 全部相同则得分
    if all(x == num for x in change_list):
        return num

    return 0


def get_line_potential_all(score_list: list, length: int) -> int:
    # 得到一行潜力，score_list为该行，length为有效长度
    return get_line_potential(score_list) * length


def get_line_score_all(score_list: list, length: int) -> int:
    # 得到一行分数，score_list为该行，length为有效长度
    return get_line_score(score_list) * length


all_cards = []
for i in [8, 4, 3]:
    for j in [9, 5, 1]:
        for k in [7, 6, 2]:
            all_cards.append(Card(i, j, k))
all_cards.append(Card(-1, -1, -1))


class Board:

    def __init__(self) -> None:
        # matrix为主矩阵，bin为垃圾箱
        self.matrix = [[Card() for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if MP[i][j] == -1:
                    self.matrix[i][j] = Card(-1, -1, -1)
        self.bin = Card()

    def get_actions(self):
        ret = []
        if self.bin.is_empty():
            ret.append(0)
        for i in range(1, 20):
            x, y = move_to_location(i)
            if self.matrix[x][y].is_empty():
                ret.append(i)
        return ret

    # 1  4  8  *  *
    # 2  5  9  13 *
    # 3  6  10 14 17
    # *  7  11 15 18
    # *  *  12 16 19

    # def is_empty(self, move: int) -> bool:
    #     i, j = self.move_to_location(move)
    #     return self.matrix[i][j].is_empty()

    def do_move(self, piece: Card, move: int) -> None:
        # 先得到坐标i, j
        i, j = move_to_location(move)
        # print(move, i, j)
        if i == j == -1:
            _piece = self.bin
        else:
            _piece = self.matrix[i][j]
        if not _piece.is_empty():
            raise Exception("位置已被占用")
        _piece.num = piece.num

    # 1  4  8  *  *
    # 2  5  9  13 *
    # 3  6  10 14 17
    # *  7  11 15 18
    # *  *  12 16 19

    # 获得分数
    def get_score(self) -> int:
        score = 0

        # 往右下
        for x, y in [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0]]:
            mx = max(x, y)
            score_list0 = [self.matrix[x + k][y + k].num[0]
                           for k in range(5 - mx)]
            score += get_line_score_all(score_list0, 5 - mx)

        # 往下
        for y in range(5):
            score_list1 = [self.matrix[k][y].num[1] for k in range(5)]
            score += get_line_score_all(score_list1, 5 - abs(y - 2))

        # 往右
        for x in range(5):
            score_list2 = [self.matrix[x][k].num[2] for k in range(5)]
            score += get_line_score_all(score_list2, 5 - abs(x - 2))

        score += self.bin.get_score()

        return score

    # 获得潜力
    def get_potential(self) -> int:
        potential = 0

        # 往右下
        for x, y in [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0]]:
            mx = max(x, y)
            score_list0 = [self.matrix[x + k][y + k].num[0]
                           for k in range(5 - mx)]
            potential += get_line_potential_all(score_list0, 5 - mx)

        # 往下
        for y in range(5):
            score_list1 = [self.matrix[k][y].num[1] for k in range(5)]
            potential += get_line_potential_all(score_list1, 5 - abs(y - 2))

        # 往右
        for x in range(5):
            score_list2 = [self.matrix[x][k].num[2] for k in range(5)]
            potential += get_line_potential_all(score_list2, 5 - abs(x - 2))

        return potential

    def show(self, title='', sub_title='') -> None:
        image = Image.new('RGB', (7 * 64, 7 * 64), (255, 255, 255))
        walls = [[[0] * 3 for i in range(9)] for j in range(9)]

        # 往右下
        for x, y in [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0]]:
            mx = max(x, y)
            score_list0 = [self.matrix[x + k][y + k].num[0]
                           for k in range(5 - mx)]
            if get_line_score(score_list0):
                for k in range(9 - mx):
                    walls[x + k][y + k][0] = 1

        # 往下
        for y in range(5):
            score_list1 = [self.matrix[k][y].num[1] for k in range(5)]
            if get_line_score(score_list1):
                for k in range(9):
                    walls[k][y + 2][1] = 1

        # 往右
        for x in range(5):
            score_list2 = [self.matrix[x][k].num[2] for k in range(5)]
            if get_line_score(score_list2):
                for k in range(9):
                    walls[x + 2][k][2] = 1

        for i in range(9):
            for j in range(9):
                try:
                    if i < 2 or j < 2:
                        raise Exception('')
                    move = location_to_move((i - 2, j - 2))
                    piece = self.matrix[i - 2][j - 2]
                    image2 = piece.get_image(move)
                except:
                    # print(i, j)
                    image2 = Image.open(
                        'resource/wall_' + str(walls[i][j][0]) + str(walls[i][j][1]) + str(walls[i][j][2]) + '.png')
                pos = (j * 64 - 64, i * 64 - j * 32 + 64)
                image.paste(image2, pos)

        image2 = Image.open('resource/wall_half.png')
        for i in range(0, 128 * 4, 128):
            image.paste(image2, (i, 0))
            image.paste(image2, (i, 7 * 64 - 32))

        image2 = self.bin.get_image(0)
        image.paste(image2, (5 * 64, 0))

        # image.show(title)
        image3 = Image.new('RGB', (7 * 64, 8 * 64), (255, 255, 255))
        image3.paste(image, (0, 64))

        draw = ImageDraw.Draw(image3)
        draw.text((10, 20), title, (0, 0, 0))
        draw.text((10, 32), sub_title, (0, 0, 0))
        image3.show()

    def get_card(seed: str):
        ret = subprocess.Popen(
            r"source\CombSeedFinder.exe",
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ret.stdin.write((seed + '\n').encode())
        ret.stdin.flush()
        ret.stdin.write(b'end\n')
        ret.stdin.flush()

        cards_str = [line.decode('utf-8').rstrip().replace('癞子', '-1,-1,-1')
                     for line in ret.stdout.readlines()][1:-1]
        cards = []
        for card_str in cards_str:
            temp = [int(i) for i in card_str.split(',')]
            cards.append(Card(temp[0], temp[1], temp[2]))
        return cards

    def get_random_cards():
        cards = all_cards * 2
        random.shuffle(cards)
        if any(card.is_wild() for card in cards[:20]):
            return cards[:20]
        else:
            return cards[20:40]

# # 1  4  8  *  *
# # 2  5  9  13 *
# # 3  6  10 14 17
# # *  7  11 15 18
# # *  *  12 16 19
