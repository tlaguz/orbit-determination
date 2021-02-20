from typing import NamedTuple

import numpy as np

from brutezerofinder import BruteZeroFinder
from consts import Consts
from earthposition import EarthPosition
from observations import Observations, ObservationPoint


class IcrsPoint(NamedTuple):
    observation: ObservationPoint
    earth_position: EarthPosition

    lambda_vector: np.array
    R_vector: np.array
    rho_vector: np.array
    r_vector: np.array

    psi_angle: float
    phi_angle: float
    n1: float
    n3: float


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

    #  Returns absolute value of arbitrary, auxiliary variable N
    def abs_N(self, ijk):
        (i, j, k) = ijk
        obsj = self.observations.points[j]
        R2 = self.earth_position.get_position(obsj.timestamp).R()

        psi2 = self.psi_angle(j)
        return np.sqrt(R2 ** 2 * np.sin(psi2) ** 2 + (R2 * np.cos(psi2) - self.A_aux(ijk)) ** 2)

    #  Returns value in radians of arbitrary, auxiliary angle \alpha
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

    #  Returns value of the gauss equation for three given observations i, j, k.
    #    phi is the equation's argument and marq and alphaarg can be provided externally for optimization purposes.
    def gauss_equation(self, ijk, phi, marg=None, alphaarg=None):
        m = marg or self.m_aux(ijk)
        alpha = alphaarg or self.alpha(ijk)

        return m * np.sin(phi) ** 4 - np.sin(phi - alpha)

    #  Returns zeroes of the gauss equation defined by `gauss_equation` method
    def solve_gauss_equation(self, ijk):
        m = self.m_aux(ijk)
        alpha = self.alpha(ijk)
        return BruteZeroFinder.find_zeroes(0, np.pi, 10**-5, lambda x: self.gauss_equation(ijk, x, m, alpha))

    #  Returns angle values in radians between Sun and Earth (Sun-Object-Earth) for three given observations i, j, k.
    #    The angle values are for observation j
    def phi_angle(self, ijk):
        (i, j, k) = ijk
        solutions = self.solve_gauss_equation(ijk)

        # removing unphysical solution
        drop = 0
        atvalue = np.pi-self.psi_angle(j)
        diff = abs(solutions[0] - atvalue)
        for i in range(1, len(solutions)):
            ndiff = abs(solutions[1] - atvalue)
            if ndiff < diff:
                drop = i
                diff = ndiff

        del solutions[drop]

        return solutions

    def position(self, ijk, phi2_angle):
        (i, j, k) = ijk

        obsi = self.observations.points[i]
        obsj = self.observations.points[j]
        obsk = self.observations.points[k]

        earthposi = self.earth_position.get_position(obsi.timestamp)
        earthposj = self.earth_position.get_position(obsj.timestamp)
        earthposk = self.earth_position.get_position(obsk.timestamp)

        tt1 = obsi.timestamp.tt
        tt2 = obsj.timestamp.tt
        tt3 = obsk.timestamp.tt
        tau1 = tt2 - tt1
        tau3 = tt3 - tt2

        lambda1_vector = self.lambda_vector(i)
        lambda2_vector = self.lambda_vector(j)
        lambda3_vector = self.lambda_vector(k)

        rho2 = earthposj.R()*(np.sin(self.psi_angle(j) + phi2_angle))/(np.sin(phi2_angle))
        rho2_vector = rho2*lambda2_vector
        r2_vector = rho2_vector + earthposj.R_vector()
        r2 = np.linalg.norm(r2_vector)

        n1 = (tau1)/(tau3+tau1)*(1+(Consts.GM*(tau3**2 + 2*tau3*tau1))/(6*r2 ** 3))
        n3 = (tau3)/(tau3+tau1)*(1+(Consts.GM*(tau1**2 + 2*tau3*tau1))/(6*r2 ** 3))

        R_vecor = n3*earthposi.R_vector() + n1*earthposk.R_vector() - earthposj.R_vector()
        Lambda_matrix = np.array([lambda1_vector, lambda2_vector, lambda3_vector])
        q_vector = np.dot(np.linalg.inv(np.transpose(Lambda_matrix)), R_vecor)

        rho1 = -q_vector[0] / n3
        rho1_vector = rho1 * lambda1_vector
        rho3 = -q_vector[2] / n1
        rho3_vector = rho3 * lambda3_vector

        r1_vector = rho1_vector + earthposi.R_vector()
        r3_vector = rho3_vector + earthposk.R_vector()

        phi1_angle = np.arcsin(earthposi.R()/np.linalg.norm(r1_vector) * np.sin(self.psi_angle(i)))
        phi3_angle = np.arcsin(earthposk.R() / np.linalg.norm(r3_vector) * np.sin(self.psi_angle(k)))

        result = []

        result.append(IcrsPoint(
            obsi,
            earthposi,
            lambda1_vector,
            earthposi.R_vector(),
            rho1_vector,
            r1_vector,
            self.psi_angle(i),
            phi1_angle,
            n1,
            n3
        ))

        result.append(IcrsPoint(
            obsj,
            earthposj,
            lambda2_vector,
            earthposj.R_vector(),
            rho2_vector,
            r2_vector,
            self.psi_angle(j),
            phi2_angle,
            n1,
            n3
        ))

        result.append(IcrsPoint(
            obsk,
            earthposk,
            lambda3_vector,
            earthposk.R_vector(),
            rho3_vector,
            r3_vector,
            self.psi_angle(k),
            phi3_angle,
            n1,
            n3
        ))

        return result

    #  Returns complete solution sets of position, for three given observations i, j, k
    #    i.e. array of possible solutions where each one is a position for i, j, k observations respectively
    def positions(self, ijk):
        result = []

        phi_angles = self.phi_angle(ijk)
        for phi in phi_angles:
            result.append(self.position(ijk, phi))

        return result

