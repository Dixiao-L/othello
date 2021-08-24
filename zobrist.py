from random import randrange

swap_side = [randrange(1 << 32), randrange(1 << 32)]
zarr = [[],[],[]]

for pn in range(64):
    zarr[0].append([randrange(1 << 32), randrange(1 << 32)])
    zarr[1].append([randrange(1 << 32), randrange(1 << 32)])
    zarr[2].append(
        [zarr[0][pn][0] ^ zarr[1][pn][0], zarr[0][pn][1] ^ zarr[1][pn][1]]
    )

def zobrist_swap(key):
    key[0] ^= swap_side[0]
    key[1] ^= swap_side[1]

def zobrist_set(key, pc, pn):
    key[0] ^= zarr[pc][pn][0]
    key[1] ^= zarr[pc][pn][1]
