class HashTable:
    def __init__(self):
        self.HASH_SIZE = (1 << 19) - 1
        self.data = [None] * (1 << 19)

    def set(self, key, eva, depth, flag, best):
        keyb = key[0] & self.HASH_SIZE
        phashe = self.data[keyb]
        if phashe == None:
            self.data[keyb] = {}
            phashe = self.data[keyb]
        elif phashe['key'] == key[1] and phashe['depth'] > depth:
            return
        phashe['key'] = key[1]
        phashe['eva'] = eva
        phashe['depth'] = depth
        phashe['flag'] = flag
        phashe['best'] = best

    def get(self, key, depth, alpha, beta):
        phashe = self.data[key[0] & self.HASH_SIZE]
        if (phashe == None):
            return False
        if (phashe['key'] != key[1]) or (phashe['depth'] < depth):
            return False
        if phashe['flag'] == 0:
            return phashe['eva']
        elif phashe['flag'] == 1:
            if phashe['eva'] <= alpha:
                return phashe['eva']
            return False
        elif phashe['flag'] == 2:
            if phashe['eva'] >= beta:
                return phashe['eva']
            return False

    def getBest(self, key):
        phashe = self.data[key[0] & self.HASH_SIZE]
        if (phashe == None):
            return None
        if phashe['key'] != key[1]:
            return None
        return phashe['best']
