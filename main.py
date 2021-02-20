#!/usr/bin/python3
import calendar
import math

import matplotlib.pyplot as plt

import numpy as np

from brutezerofinder import BruteZeroFinder
from earthposition import EarthPosition
from icrsposition import IcrsPosition
from observations import Observations

earthPosition = EarthPosition("earth.csv")
observations = Observations("pallas.csv")

icrsposition = IcrsPosition(earthPosition, observations)
phi = icrsposition.phi_angle((0, 1, 2)) #[0]*360/(2*math.pi)

r = list(calendar.month_abbr)

pass