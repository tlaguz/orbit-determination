import numpy as np

from consts import Consts
from earthposition import EarthPosition
from observations import Observations


class IcrsPosition:
    earth_position: EarthPosition
    observations: Observations

    def __init__(self, earth_position: EarthPosition, observations: Observations):
        self.earth_position = earth_position
        self.observations = observations

    #  Returns vector oriented in direction to object from Earth for given observation number n
    def lambda_vector(self, n):
        obs = self.observations.points[n]
        return np.array([
            np.cos(obs.DEC) * np.cos(obs.RA),
            np.cos(obs.DEC) * np.sin(obs.RA),
            np.sin(obs.DEC)
        ])

    #  Returns angle value in radians between Sun and the object for given observation number n
    def psi_angle(self, n):
        obs = self.observations.points[n]
        earthpos = self.earth_position.get_position(obs.timestamp)

        l = self.lambda_vector(n)
        cosine_psi = (earthpos.X * l[0] + earthpos.Y * l[1] + earthpos.Z * l[2])/earthpos.R()

        return np.arccos(cosine_psi)

    #  Returns determinant of a matrix built from vectors: [\lambda_i, \lambda_j, \lambda_k] for observations i, j, k
    def lambda_det(self, ijk):
        (i, j, k) = ijk
        matrix = np.matrix([self.lambda_vector(i), self.lambda_vector(j), self.lambda_vector(k)])
        return np.linalg.det(matrix)

    #  Returns determinant of a matrix built from vectors: [\lambda_i, R_n, \lambda_k] for observations i, j, k,
    #    where n: {0, 1, 2} -> {i, j, k}
    def d_det(self, ijk, n):
        (i, j, k) = ijk
        obs = self.observations.points[i+n]
        earthpos = self.earth_position.get_position(obs.timestamp)

        matrix = np.matrix([self.lambda_vector(i), np.array([earthpos.X, earthpos.Y, earthpos.Z]), self.lambda_vector(k)])
        return np.linalg.det(matrix)

    #  Returns auxiliary variable A
    def A(self, ijk):
        (i, j, k) = ijk
        tt1 = self.observations.points[i].timestamp.tt
        tt3 = self.observations.points[k].timestamp.tt

        return 1.0/self.lambda_det(ijk) * (
                self.d_det(ijk, 1) * (tt3) / (tt3+tt1) +
                self.d_det(ijk, 3) * (tt1) / (tt3 + tt1) -
                self.d_det(ijk, 2)
        )

    #  Returns auxiliary variable B
    def B(self, ijk):
        (i, j, k) = ijk
        tt1 = self.observations.points[i].timestamp.tt
        tt3 = self.observations.points[k].timestamp.tt

        return 1.0 / self.lambda_det(ijk) * (
                self.d_det(ijk, 1) * (tt3) / (tt3 + tt1) * (Consts.GM * (2 * tt3 * tt1 + tt1**2)) / (6) +
                self.d_det(ijk, 3) * (tt1) / (tt3 + tt1) * (Consts.GM * (2 * tt3 * tt1 + tt3**2)) / (6)
        )

    def abs_N(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        psi2 = self.psi_angle(j)
        return np.sqrt(R2**2 * np.sin(psi2)**2 + (R2 * np.cos(psi2) - self.A(ijk))**2)

    def alpha(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        psi2 = self.psi_angle(j)
        absN = self.abs_N(ijk)
        sinalpha = -R2 * np.sin(psi2) / absN
        cosalpha = (R2 * np.cos(psi2) - self.A(ijk)) / absN

        return np.arctan(sinalpha / (1+cosalpha)) * 2

    #  Returns auxiliary variable m
    def m(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        return (-self.B(ijk) * np.sin(self.alpha(ijk)))/(R2**4 * np.sin(self.psi_angle(j))**4)

    def gauss_equation(self, ijk, phi):
        (i, j, k) = ijk

        return self.m(ijk) * np.sin(phi)**4 - np.sin(phi - self.alpha(ijk))

