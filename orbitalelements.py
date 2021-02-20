from typing import NamedTuple

import numpy as np
from numpy import math

from brutezerofinder import BruteZeroFinder
from consts import Consts
from earthposition import EarthPosition
from icrsposition import IcrsPosition, IcrsPoint
from observations import Observations, ObservationPoint


class Elements(NamedTuple):
    inclination: float
    Omega: float
    e: float
    tp: float
    a: float
    omega: float


class OrbitalElements:
    obs: [IcrsPoint]

    def __init__(self,
                 obs: [IcrsPoint]):
        self.obs = obs

    z_vector = np.array([0, 0, 1])
    x_vector = np.array([1, 0, 0])

    def n_vec(self):
        obs = self.obs
        cross = np.cross(obs[0].r_vector, obs[2].r_vector)
        return cross / (np.linalg.norm(cross))

    def inclination(self):
        return np.arccos(np.dot(self.n_vec(), self.z_vector))

    def w_vector(self):
        cross = np.cross(self.z_vector, self.n_vec())
        return (cross) / (np.linalg.norm(cross))

    def Omega(self):
        sinOmega = np.dot(self.z_vector, np.cross(self.x_vector, self.w_vector()))
        cosOmega = np.dot(self.x_vector, self.w_vector())

        return np.arctan(sinOmega / (1 + cosOmega)) * 2

    def parameters(self):
        r1 = np.linalg.norm(self.obs[0].r_vector)
        r2 = np.linalg.norm(self.obs[1].r_vector)
        r3 = np.linalg.norm(self.obs[2].r_vector)

        r1vec = self.obs[0].r_vector/r1
        r2vec = self.obs[1].r_vector/r2
        r3vec = self.obs[2].r_vector/r3

        A = np.dot(r2vec, r1vec)
        B = np.dot(r3vec, r1vec)
        C = np.dot(np.cross(r1vec, r2vec), self.n_vec())
        D = np.dot(np.cross(r1vec, r3vec), self.n_vec())

        # p = a(1-e^2) ; a -
        p = -((-C * r1 * r2 * r3 +
              B * C * r1 * r2 * r3 +
              D * r1 * r2 * r3 +
              -A * D * r1 * r2 * r3)
             / (C * r1 * r2 +
                -D * r1 * r3 +
                -B * C * r2 * r3 +
                A * D * r2 * r3
                ))

        # S1=e sin(f_1)
        S1 = -((-r1 * r2 +
               A * r1 * r2 +
               r1 * r3 +
               -B * r1 * r3 +
               -A * r2 * r3 +
               B * r2 * r3)
              / (C * r1 * r2 +
                 -D * r1 * r3 +
                 -B * C * r2 * r3 +
                 A * D * r2 * r3))

        # C1=e cos(f_1)
        C1 = -((C * r1 * r2 +
               -D * r1 * r3 +
               -C * r2 * r3 +
               D * r2 * r3)
              / (C * r1 * r2 +
                 -D * r1 * r3 +
                 -B * C * r2 * r3 +
                 A * D * r2 * r3))

        e = (S1**2 + C1**2)**(1/2)
        anomaly = np.arctan(S1/(e+C1))*2
        a = (p)/(1 - e**2)

        cos_omega_plus_anomaly = np.dot(self.w_vector(), r1vec)
        sin_omega_plus_anomaly = np.dot(np.cross(self.w_vector(), r1vec), self.n_vec())

        omega = np.arctan(sin_omega_plus_anomaly/(1+cos_omega_plus_anomaly))*2 - anomaly

        return (e, anomaly, a, omega)

    def u(self, e, anomaly):
        return 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(anomaly/2))

    def tp(self, e, a, anomaly):
        n = np.sqrt(Consts.GM/a**3)
        u = self.u(e, anomaly)
        t1 = self.obs[0].observation.timestamp.tt
        return (u - e*np.sin(u) - n*t1)/(-n)

    def elements(self):
        (e, anomaly, a, omega) = self.parameters()
        tp = self.tp(e, a, anomaly)
        return Elements(
            self.inclination(),
            self.Omega(),
            e,
            tp,
            a,
            omega
        )
