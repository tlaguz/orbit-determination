import calendar
from datetime import date


def leap_years_before(year):
    year = year-1
    return int((year / 4) - (year / 100) + (year / 400))


jd2001 = 2451910.50000  # julian date for 1 Jan 2001 00:00:00 (2000 was leap)


class Timestamp:
    # time in TT representation
    tt: float

    def __init__(self, tt: float):
        self.tt = tt

    @staticmethod
    def create_from_ut(ut):
        ut = ut.strip()
        Y = int(ut.split(" ")[0].split("-")[0])
        M = ut.split(" ")[0].split("-")[1]
        M = list(calendar.month_abbr).index(M)
        D = int(ut.split(" ")[0].split("-")[2])
        HH = int(ut.split(" ")[1].split(":")[0])
        MM = int(ut.split(" ")[1].split(":")[1])

        leapYears = leap_years_before(Y) - leap_years_before(2001)
        normalYears = Y-2001-leapYears

        daysInY = (date(Y, M, D) - date(Y, 1, 1)).days

        jd = jd2001 + 365*normalYears + 366*leapYears + daysInY + (HH)/24.0 + MM/(24.0*60.0)

        return Timestamp.create_from_jd(jd)

    @staticmethod
    def create_from_jd(jd):
        tt = jd + 32.184/(24*60*60)
        tt = tt + 37/(24*60*60)  # leap seconds after 31 dec 2016 @todo implement for earlier dates
        return Timestamp.create_from_tt(tt)

    @staticmethod
    def create_from_tt(tt):
        return Timestamp(float(tt))
