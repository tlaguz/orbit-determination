#!/usr/bin/python3
import calendar

from earthposition import EarthPosition
from icrsposition import IcrsPosition
from observations import Observations

earthPosition = EarthPosition("earth.csv")
observations = Observations("pallas.csv")

icrsposition = IcrsPosition(earthPosition, observations)

psi1 = icrsposition.psi_angle(0)
psi2 = icrsposition.psi_angle(1)
psi3 = icrsposition.psi_angle(2)
psi4 = icrsposition.psi_angle(3)
psi5 = icrsposition.psi_angle(4)

det = icrsposition.lambda_det((2, 3, 4))
ddet = icrsposition.d_det((2, 3, 4), 2)

r = list(calendar.month_abbr)

pass