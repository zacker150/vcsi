from collections import namedtuple
from enum import Enum

import numpy as np

TimestampPosition = Enum('TimestampPosition', "north south east west ne nw se sw center")
print("timestamp")

class Color(namedtuple('Color', ['r', 'g', 'b', 'a'])):

    @staticmethod
    def to_hex(component):
        h = hex(component).replace("0x", "").upper()
        return h if len(h) == 2 else "0" + h

    def __str__(self):
        return "".join([self.to_hex(x) for x in [self.r, self.g, self.b, self.a]])


class Frame:
    def __init__(self, filename: str, blurriness: np.float64, timestamp: int, avg_color: Color):
        self.filename = filename
        self.blurriness = blurriness
        self.timestamp = timestamp
        self.avg_color = avg_color


class Grid:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return "%sx%s" % (self.x, self.y)
