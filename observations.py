import math
from typing import NamedTuple

from timestamp import Timestamp


def degangle_to_radians(angle: str):
    s = angle.split(" ")
    sign = 1 if int(s[0]) >= 0 else -1

    return (2.0*math.pi)/360.0 * (abs(int(s[0])) + int(s[1])/60.0 + float(s[2])/3600.0) * sign


def hourangle_to_radians(angle: str):
    s = angle.split(" ")

    return (2.0 * math.pi) / 24.0 * (abs(int(s[0])) + int(s[1]) / 60.0 + float(s[2]) / 3600.0)


class ObservationPoint(NamedTuple):
    timestamp: Timestamp
    RA: float  # radians
    DEC: float  # radians
    interpolated: bool


class Observations:
    points = []

    def __init__(self, fname):
        f = open(fname)
        while(f.readline() != "$$SOE\n"):
            pass

        while((r := f.readline()) != "$$EOE\n"):
            s = r.split(",")
            self.points.append(ObservationPoint(
                Timestamp.create_from_ut(s[0]),
                float(hourangle_to_radians(s[3])),
                float(degangle_to_radians(s[4])),
                False
            ))

        f.close()
