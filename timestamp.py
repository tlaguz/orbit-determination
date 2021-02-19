

class Timestamp:
    # time in TT representation
    time: float

    def __init__(self, tt: float):
        self.time = tt

    @staticmethod
    def create_from_ut(ut):
        # todo
        return Timestamp()

    @staticmethod
    def create_from_tt(tt):
        return Timestamp(float(tt))
