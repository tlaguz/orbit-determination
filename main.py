#!/usr/bin/python3
import calendar

from earthposition import EarthPosition
from observations import Observations

earthPosition = EarthPosition("earth.csv")
observations = Observations("pallas.csv")

r = list(calendar.month_abbr)

pass