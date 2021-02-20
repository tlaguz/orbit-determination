
import numpy as np

#  Bruteforce zero finder
class BruteZeroFinder:
    @staticmethod
    def find_zeroes(start, stop, step, fn):
        zeroes = []
        curr = start
        prevsign = np.sign(fn(start))
        while(curr <= stop):
            currsign = np.sign(fn(curr))
            if(prevsign != currsign):
                zeroes.append((curr + curr-step)/2)

            prevsign = currsign
            curr += step

        return zeroes
