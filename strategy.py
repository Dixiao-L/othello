from copy import deepcopy
import sys
import time
import math
from zobrist import zobrist_swap
import hash_table
from random import randrange
from collections import defaultdict
from typing import List, Tuple
from map import new_mape
from map import history
from map import mape
from debug import debug

weight = [6, 11, 2, 10, 3] # corner, steady, frontier, mobility, parity

hash = hash_table.HashTable()

rnd = [
		{'s':  0, 'a':  1, 'b':  8, 'c':  9, 'dr': [ 1,  8]},
		{'s':  7, 'a':  6, 'b': 15, 'c': 14, 'dr': [-1,  8]},
		{'s': 56, 'a': 57, 'b': 48, 'c': 49, 'dr': [ 1, -8]},
		{'s': 63, 'a': 62, 'b': 55, 'c': 54, 'dr': [-1, -8]}
	]

class MTD_ai:
    def __init__(self) -> None:
        self.TIME_LIMIT = 2.975
        self.OUTCOME_DEPTH = 8  # 终局搜索深度
        self.OUTCOME_COARSE = 10 # 终局模糊搜索深度
        self.start_time = 0
        self.out_time = 0
        self.max_depth = 0

    def startSearch(self, mape) -> int:
        self.start_time = time.time()
        debug(f"Start Searching Timestamp: {self.start_time}")
        f = 0
        if mape.space <= self.OUTCOME_DEPTH: # 终局搜索
            self.out_time = self.start_time + 2.975
            self.max_depth = mape.space
            try:
                if self.max_depth >= self.OUTCOME_COARSE:
                    f = self.alpha_beta(mape, self.max_depth, -math.inf, math.inf)
                else:
                    f = self.mtd_f(mape, self.max_depth, f)
            except TimeoutError:
                debug("Final Timeout")
            debug(f"Final Search result: {self.max_depth} {mape.space} {mape.player} {f * (1 - 2 * mape.player)}")
            return hash.getBest(mape.key)

        self.out_time = self.start_time + self.TIME_LIMIT
        self.max_depth = 0
        try:
            while self.max_depth < mape.space:
                self.max_depth += 1
                f = self.mtd_f(mape, self.max_depth, f)
                best = hash.getBest(mape.key) # error: return prev
                debug(f"{self.max_depth} {f * (1 - 2 * mape.player)} {best}")
        except TimeoutError:
            debug("Timeout")
        debug(f"Search result: {self.max_depth - 1} {mape.space} {(1 - 2 * mape.player)} {f * (1 - 2 * mape.player)}")
        debug(f"result timestamp: {time.time()}")
        return best

    def evaluate(self, mape): # 评估函数
        def map_value(n: int):
            if n >= 64:
                return None
            l = mape.board[n]
            if l == 2:
                return 0
            elif l == 0:
                return 1
            else:
                return -1

        corner = 0
        steady = 0
        uk = defaultdict(lambda: 0)
        for v in rnd:
            if map_value(v['s']) == 0:
                corner += map_value(v['a']) * -3   # side-star
                corner += map_value(v['b']) * -3   # side-star
                corner += map_value(v['c']) * -6   # star
                continue
            corner += map_value(v['s']) * 15   # corner
            steady += map_value(v['s'])

            for k in range(2):
                if uk[
                    v['s'] + v['dr'][k]
                ]: continue
                eb = True
                tmp = 0
                for j in range(1, 8):
                    t = map_value(
                        v['s'] + v['dr'][k] * j
                    )
                    if t == 0: break
                    elif eb and (t == map_value(v['s'])):
                        steady += t
                    else:
                        eb = False
                        tmp += t
                if (j == 7) and \
                    map_value(v['s'] + v['dr'][k] * 7) != 0:
                    steady += tmp
                    uk[
                        v['s'] + v['dr'][k] * 6
                    ] = True

        frontier = 0
        i = 8
        while i <= 54:
            i += 3 if i & 7 == 6 else 1
            if map_value(i) == 0: continue
            for j in range(8):
                if map_value(mape.dire(i, j)) == 0:
                    frontier -= map_value(i)
                    break

        mobility = (mape.nextNum - mape.prevNum) * (1 - 2 * mape.player)

        if mape.space < 18:
            if mape.space % 2 == 0:
                parity = -1 + 2 * mape.player
            else:
                parity = 1 - 2 * mape.player
        else:
            parity = 0

        rv = corner * weight[0] \
            + steady * weight[1] \
            + frontier * weight[2] \
            + mobility * weight[3] \
            + parity * weight[4]

        return rv * (1 - 2 * mape.player)

    def outcome(self, mape): # 终局结果
        s = mape.black - mape.white
        if self.max_depth >= self.OUTCOME_COARSE:
            return (sgn(s) << 14) * (1 - 2 * mape.player)
        return ((s + mape.space * sgn(s)) << 14) * (1 - 2 * mape.player)

    def alpha_beta(self, mape, depth: int, alpha, beta):
        # debug(f"alpha-beta time: {time.time()}")

        if time.time() > self.out_time:  # 超时
            # debug("Timeout")
            raise TimeoutError

        hv = hash.get(mape.key, depth, alpha, beta)
        # debug(f"hash.get time: {time.time()}")
        if hv != False:
            return hv

        if mape.space == 0:
            return self.outcome(mape)

        # mape.allow_location()
        # debug(f"allow_location: {mape.nextIndex}")

        if mape.nextNum == 0:
            if mape.prevNum ==0:
                return self.outcome(mape)
            mape.pass_round()
            return -self.alpha_beta(mape, depth, -beta, -alpha)

        # debug(f"depth: {depth} {type(depth)}")
        if depth <= 0:
            e = self.evaluate(mape)
            hash.set(mape.key, e, depth, 0, None)
            # debug("hash.set")
            return e

        hd = hash.getBest(mape.key)
        if hd != None:
        # if hd in mape.nextIndex:
            # debug(mape.nextIndex)
            move_to_head(mape.nextIndex, hd)

        hist = history[mape.player][mape.space]
        hashf = 1   # 最佳估值类型 0->precise 1->alpha 2->beta
        best_val = -math.inf
        best_act = None
        for n in mape.nextIndex:
            v = -self.alpha_beta(new_mape(mape, n), depth - 1, -beta, -alpha)
            if v > best_val:
                best_val = v
                best_act = n
                if v > alpha:
                    alpha = v
                    hashf = 0
                    move_up(hist, n)
                if v >= beta:
                    hashf = 2
                    break   # cutoff

        # debug(f"hist: {len(hist)}")
        move_to_head(hist, best_act)
        hash.set(mape.key, best_val, depth, hashf, best_act)
        return best_val

    def mtd_f(self, mape, depth: int, f):
        upper_bound = math.inf
        lower_bound = -math.inf
        while True:
            beta = f + 1 if f == lower_bound else f
            f = self.alpha_beta(mape, depth, beta - 1, beta)
            if f < beta:
                upper_bound = f
            else:
                lower_bound = f
            if lower_bound >= upper_bound:
                break
        if f < beta:
            f = self.alpha_beta(mape, depth, f - 1, f)
        return f

    def run(self, mape) -> int:
        if mape.nextNum == 0:
            mape.pass_round()
            return -1
        elif mape.nextNum == 1:
            return mape.nextIndex[0]
        elif mape.space <= 58:
            return self.startSearch(mape)
        else:
            return mape.nextIndex[randrange(mape.nextNum)]

def sgn(num):   # sign function
    if num < 0:
        return -1
    elif num > 0:
        return 1
    else:
        return 0

def move_to_head(arr, n):   # 返回头部节点
    if arr[0] == n:
        return
    if n in arr:
        arr.remove(n)
    else:
        arr = arr[:-1]
    arr.insert(0, n)

def move_up(arr, n):    # 返回父节点
    if arr[0] == n:
        return
    if n in arr:
        i = arr.index(n)
        arr[i], arr[i - 1] = arr[i - 1], n

def debug(s):
    print(s, file=sys.stderr, flush=True)

def reversi_ai(player: int, board: List[int], allow: List[bool], prev_map):
    if prev_map == None:    # first step
        prev_map = mape(player, board, allow)
    else:
        k = []
        for i in range(64):
            if prev_map.board[i] == 2 and board[i] != 2:    # 对方落子
                k.append(i)
        if len(k) == 0:  # 对方未落子
            debug("beside pass")
            prev_map.pass_round()
            # debug(f"{prev_map.player}, {player}")
        elif len(k) == 1:    # 对方落1子
            prev_map = new_mape(prev_map, k[0])
        else:
            pass_map = mape(player, board, allow)
            pass_map.key = deepcopy(prev_map.key)
            for i in k:
                zobrist_swap(pass_map.key)
            prev_map = pass_map


    # debug("ai init")
    # if prev_map != None:
    #     for i in range(64):
    #         if prev_map.board[i] == 2 and board[i] != 2:
    #             n = i
    #             prev_map = new_mape(prev_map, n)
    #             break
    #     new_map = mape(player, board, allow, prev_map.prevNum, prev_map.key)
    # else:
    #     new_map = mape(player, board, allow, 0, [0, 0])

    # debug("map stored:")
    # debug(f"nextNum:{new_map.nextNum}")
    # debug(f"space:{new_map.space}")

    # ai 走棋
    agent = MTD_ai()
    # debug("agent deployed")

    v = agent.run(prev_map)
    if v != -1:
        prev_map = new_mape(prev_map, v)
    else:
        prev_map.pass_round()
        v = 0
    # debug(f"frontier: {prev_map.frontier}")

    debug(f"v: {v}")
    # debug(f"supposed next index: {prev_map.nextIndex}")


    x, y = divmod(v, 8)
    # debug(f"x:{x}, y:{y}")

    # prev_map 返回下完的棋盘

    return x, y, prev_map
