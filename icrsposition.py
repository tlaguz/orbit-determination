import numpy as np

from brutezerofinder import BruteZeroFinder
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
        vec = np.array([
            np.cos(obs.DEC) * np.cos(obs.RA),
            np.cos(obs.DEC) * np.sin(obs.RA),
            np.sin(obs.DEC)
        ])

        angle = -2*np.pi/360 * (23 + 26/60 + 14/3600)
        rotmatrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        res = np.dot(rotmatrix, vec)
        return res

    #  Returns angle value in radians between Sun and the object for given observation number n
    def psi_angle(self, n):
        obs = self.observations.points[n]
        earthpos = self.earth_position.get_position(obs.timestamp)

        l = self.lambda_vector(n)
        cosine_psi = -(earthpos.X * l[0] + earthpos.Y * l[1] + earthpos.Z * l[2])/earthpos.R()

        return np.arccos(cosine_psi)

    #  Returns determinant of a matrix built from vectors: [\lambda_i, \lambda_j, \lambda_k] for observations i, j, k
    def lambda_det(self, ijk):
        (i, j, k) = ijk
        matrix = np.matrix([self.lambda_vector(i), self.lambda_vector(j), self.lambda_vector(k)])
        return np.linalg.det(matrix)

    #  Returns determinant of a matrix built from vectors: [\lambda_i, R_n, \lambda_k] for observations i, j, k,
    #    where n: {1, 2, 3} -> {i, j, k}
    def d_det(self, ijk, n):
        (i, j, k) = ijk
        obs = self.observations.points[i+n-1]
        earthpos = self.earth_position.get_position(obs.timestamp)

        matrix = np.array([self.lambda_vector(i), np.array([earthpos.X, earthpos.Y, earthpos.Z]), self.lambda_vector(k)])
        return np.linalg.det(matrix)

    #  Returns auxiliary variable A
    def A_aux(self, ijk):
        (i, j, k) = ijk
        tt1 = self.observations.points[i].timestamp.tt
        tt2 = self.observations.points[j].timestamp.tt
        tt3 = self.observations.points[k].timestamp.tt

        tau1 = tt2 - tt1
        tau3 = tt3 - tt2

        return 1.0/self.lambda_det(ijk) * (
                self.d_det(ijk, 1) * (tau3) / (tau3 + tau1) +
                self.d_det(ijk, 3) * (tau1) / (tau3 + tau1) -
                self.d_det(ijk, 2)
        )

    #  Returns auxiliary variable B
    def B_aux(self, ijk):
        (i, j, k) = ijk
        tt1 = self.observations.points[i].timestamp.tt
        tt2 = self.observations.points[j].timestamp.tt
        tt3 = self.observations.points[k].timestamp.tt

        tau1 = tt2 - tt1
        tau3 = tt3 - tt2

        return 1.0 / self.lambda_det(ijk) * (
                self.d_det(ijk, 1) * (tau3) / (tau3 + tau1) * (Consts.GM * (2 * tau3 * tau1 + tau1**2)) / (6) +
                self.d_det(ijk, 3) * (tau1) / (tau3 + tau1) * (Consts.GM * (2 * tau3 * tau1 + tau3**2)) / (6)
        )

    def abs_N(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        psi2 = self.psi_angle(j)
        return np.sqrt(R2 ** 2 * np.sin(psi2) ** 2 + (R2 * np.cos(psi2) - self.A_aux(ijk)) ** 2)

    def alpha(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        psi2 = self.psi_angle(j)
        absN = self.abs_N(ijk)
        B_aux = self.B_aux(ijk)
        if (absN * B_aux < 0):
            absN = -absN

        sinalpha = -R2 * np.sin(psi2) / absN
        cosalpha = (R2 * np.cos(psi2) - self.A_aux(ijk)) / absN

        return np.arctan(sinalpha / (1+cosalpha)) * 2

    #  Returns auxiliary variable m
    def m_aux(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        return (-self.B_aux(ijk) * np.sin(self.alpha(ijk))) / (R2 ** 4 * np.sin(self.psi_angle(j)) ** 4)

    def gauss_equation(self, ijk, phi, marg = None, alphaarg = None):
        m = marg or self.m_aux(ijk)
        alpha = alphaarg or self.alpha(ijk)

        return m * np.sin(phi) ** 4 - np.sin(phi - alpha)

    #  Returns zeroes of the gauss equation defined by `gauss_equation` method
    def solve_gauss_equation(self, ijk):
        m = self.m_aux(ijk)
        alpha = self.alpha(ijk)
        return BruteZeroFinder.find_zeroes(0, np.pi, 10**-5, lambda x: self.gauss_equation(ijk, x, m, alpha))

    #  Returns angle value in radians between Sun and Earth (Sun-Object-Earth) for three given observations
    def phi_angle(self, ijk):
        (i, j, k) = ijk
        solutions = self.solve_gauss_equation(ijk)

        # removing unphysical solution
        drop = 0
        atvalue = np.pi-self.psi_angle(j)
        diff = abs(solutions[0] - atvalue)
        for i in range(1,len(solutions)):
            ndiff = abs(solutions[1] - atvalue)
            if ndiff < diff:
                drop = i
                diff = ndiff

        del solutions[drop]

        return solutions
