import random

class Zobrist:
    def __init__(self) -> None:
        self.swap_side = [random.randrange(2 << 31), random.randrange(2 << 31)]
        self.zarr = []

        for pn in range(64):
            self.zarr[0][pn] = [random.randrange(2 << 31), random.randrange(2 << 31)]
            self.zarr[1][pn] = [random.randrange(2 << 31), random.randrange(2 << 31)]
            self.zarr[2][pn] = list(i ^ i for i in self.zarr[0][pn])

    def swap(self, key):
        key[0] ^= self.swap_side[0]
        key[1] ^= self.swap_side[1]

    def set(self, key, pc, pn):
        key[0] ^= self.zarr[pc][pn][0]
        key[1] ^= self.zarr[pc][pn][1]
