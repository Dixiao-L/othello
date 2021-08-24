from debug import debug
from copy import deepcopy
from zobrist import zobrist_swap
from zobrist import zobrist_set

history = [[],[]]
for i in range(2):
    for j in range(61):
        history[i].append([0, 63, 7, 56, 37, 26, 20, 43, 19, 29, 34, 44, 21, 42, 45, 18, 2, 61, 23, 40, 5, 58, 47, 16, 10, 53, 22, 41, 13, 46, 17, 50, 51, 52, 12, 11, 30, 38, 25, 33, 4, 3, 59, 60, 39, 31, 24, 32, 1, 62, 15, 48, 8, 55, 6, 57, 9, 54, 14, 49])

class mape:
    # def __init__(self, player: int, board: List[int], allow: List[int], prevNum: int, key: List[int, int]) -> None:
    def __init__(self, player, board, allow) -> None:
        self.player = player
        self.board = board[::]

        self.frontier = []
        self.cal_frontier()

        self.nextIndex = [] # 下一步可走位置
        self.nextNum = 0    # 下一步可走位置数
        for i in range(64):
            if allow[i]:
                self.nextIndex.append(i)
                self.nextNum += 1

        # debug(f"initial nextIndex: {self.nextIndex}")
        self.nextRev = dict()   # 下一步可走位置的反转棋子
        for i in self.nextIndex:    # 计算下一步反转棋子
            self.cal_rev(i)

        self.prevNum = 0    # 上一步可走位置数
        self.key = [0, 0]      # 哈希表键值

        self.black = board.count(0)
        self.white = board.count(1)
        self.space = board.count(2)

    # def map_index(x, y):
    #     '''
    #     获取 (x,y) 点对应的下标
    #     '''
    #     return (x << 3) + y

    def dire(self, i: int, j: int):
        '''
        获取棋盘格某一方向的格子
        超过边界返回64
        '''
        dr = [-8, -7, 1, 9, 8, 7, -1, -9]
        bk = [8, 0, 0, 0, 8, 7, 7, 7]
        i += dr[j]
        return 64 if (i >= 64 or i < 0 or (i & 7) == bk[j]) else i

    def cal_frontier(self):
        for i in range(64):
            self.frontier.append(False)
            if self.board[i] == 2:
                for j in range(8):
                    k = self.dire(i, j)
                    if k == 64: continue
                    if self.board[k] != 2:
                        self.frontier[i] =True
                        break

    def cal_rev(self, i) -> bool:
        possible_rev = []
        for j in range(8):  # 8 directions
            lb = 0
            k = self.dire(i, j)
            while k != 64:
                if self.board[k] != 1 - self.player:    # match blank or self
                    break
                lb += 1
                possible_rev.append(k)
                k = self.dire(k, j) # march forward
            if k == 64 or self.board[k] != self.player: # edge or blank
                if lb != 0:
                    possible_rev = possible_rev[:-lb]
        if len(possible_rev):
            self.nextRev[i] = possible_rev[:]
            return True
        return False

    def allow_location(self) -> None:
        '''
        查找可走位置
        '''
        # debug(f"calculate location start time: {time.time()}")
        self.nextIndex = []
        self.nextRev = dict()
        self.nextNum = 0

        hist = history[self.player][self.space]
        # debug(f"type hist: {type(hist)}")

        for i in range(60):
            fin = hist[i]
            # debug(str(hist))
            if (self.frontier[fin] == False):
                continue
            if self.cal_rev(fin):
                self.nextIndex.append(fin)
                self.nextNum += 1

    def pass_round(self):
        self.player = 1 - self.player
        self.prevNum = self.nextNum
        self.allow_location()
        zobrist_swap(self.key)

def new_mape(old_map, n):
    # debug(f"new map n: {n}")
    nm = deepcopy(old_map)
    nm.board[n] = old_map.player

    nm.key = deepcopy(old_map.key)
    zobrist_set(nm.key, old_map.player, n)

    nm.frontier = old_map.frontier[::]
    nm.frontier[n] = False
    for i in range(8):
        k = old_map.dire(n, i)
        if k != 64 and nm.board[k] == 2:
            nm.frontier[k] = True

    ne = old_map.nextRev[n]
    l = 0
    for i in ne:
        nm.board[i] = old_map.player
        zobrist_set(nm.key, 2, i)
        l += 1

    # calculate space, b&w
    if old_map.player == 0:
        nm.black = old_map.black + l + 1
        nm.white = old_map.white - l
    else:
        nm.white = old_map.white + l + 1
        nm.black = old_map.black - l

    nm.space = 64 - nm.black - nm.white
    nm.player = 1 - old_map.player
    nm.prevNum = old_map.nextNum

    nm.allow_location()

    zobrist_swap(nm.key)
    return nm
