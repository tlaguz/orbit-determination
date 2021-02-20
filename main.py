#!/usr/bin/python3
import calendar
import math

import numpy as np

from brutezerofinder import BruteZeroFinder
from earthposition import EarthPosition
from icrsposition import IcrsPosition
from observations import Observations

earthPosition = EarthPosition("earth.csv")
observations = Observations("pallas.csv")

icrsposition = IcrsPosition(earthPosition, observations)

psi1 = icrsposition.psi_angle(0)*360/(2*math.pi)
psi2 = icrsposition.psi_angle(1)*360/(2*math.pi)
psi3 = icrsposition.psi_angle(2)*360/(2*math.pi)
psi4 = icrsposition.psi_angle(3)*360/(2*math.pi)
psi5 = icrsposition.psi_angle(4)*360/(2*math.pi)

phi = icrsposition.solve_gauss_equation((0, 1, 2))[0]*360/(2*math.pi)

r = list(calendar.month_abbr)

pass