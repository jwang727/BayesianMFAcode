import numpy as np
from math import log10, floor

def round_to_poweroften(x):
    #rounds number to nearest power of 10
    if x==0:
        ans=0
    else:
        ans=np.sign(x)*10 ** round(log10(abs(x))-log10(5.5) + 0.5)
    return ans

def calculatemse(predictions, truevalues):
    mse = np.sqrt(np.mean((predictions - truevalues) ** 2))
    return mse


def calculatemaxerror(predictions, truevalues):
    maxerror = np.max(np.abs(predictions - truevalues))
    return maxerror