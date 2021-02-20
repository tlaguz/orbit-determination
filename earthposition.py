from typing import NamedTuple

import numpy as np

from timestamp import Timestamp


class EarthPoint(NamedTuple):
    timestamp: Timestamp
    X: float
    Y: float
    Z: float
    VX: float
    VY: float
    VZ: float
    interpolated: bool

    def R_vector(self):
        return np.array([self.X, self.Y, self.Z])

    #  Distance between Sun and the Earth
    def R(self):
        return np.sqrt(self.X**2 + self.Y**2 + self.Z**2)


class EarthPosition:
    points: [EarthPoint]

    def __init__(self, fname):
        self.points = []
        f = open(fname)
        while(f.readline() != "$$SOE\n"):
            pass

        while((r := f.readline()) != "$$EOE\n"):
            s = r.split(",")
            self.points.append(EarthPoint(
                Timestamp.create_from_tt(s[0]),
                float(s[2]),
                float(s[3]),
                float(s[4]),
                float(s[5]),
                float(s[6]),
                float(s[7]),
                False
            ))

        f.close()

    def get_position(self, timestamp):
        point1 = 0
        point2 = 0
        # Assumes that EarthPoints is ordered ascending by time
        for i in range(len(self.points)):
            if(self.points[i].timestamp.tt > timestamp.tt):
                if(i < 1):
                    raise ValueError("Timestamp out of range for Earth position. Can't interpolate.")
                point1 = self.points[i-1]
                point2 = self.points[i]
                break

        if(point1 == 0):
            raise ValueError("Timestamp out of range for Earth position. Can't interpolate.")

        if(point1.timestamp.tt == timestamp.tt):
            return point1

        dt = point2.timestamp.tt - point1.timestamp.tt
        dt2 = timestamp.tt - point1.timestamp.tt
        return EarthPoint(
            timestamp,
            point1.X + (point2.X - point1.X) / (dt) * dt2,
            point1.Y + (point2.Y - point1.Y) / (dt) * dt2,
            point1.Z + (point2.Z - point1.Z) / (dt) * dt2,
            point1.VX + (point2.VX - point1.VX) / (dt) * dt2,
            point1.VY + (point2.VY - point1.VY) / (dt) * dt2,
            point1.VZ + (point2.VZ - point1.VZ) / (dt) * dt2,
            True
        )
