import pathlib
import pickle as pk

import matplotlib.pyplot as plt

from detect import center_hand

pkls = pathlib.Path("records/").glob("*.pkl")
pkls = list(map(str, pkls))
pkls = sorted(pkls)

plt.ion()

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

for path in pkls:
    with open(path, "rb") as out:
        frame = pk.load(out)
    frame = center_hand(frame)
    plt.pause(.01)
    ax.clear()
    ax.scatter(*frame.T)
    ax.set_xlim([-.1, 2])
    ax.set_ylim([-.1, 2])
    ax.set_zlim([-.1, 2])
