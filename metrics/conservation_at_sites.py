"""
Date: October 27th, 2019
Plot the cross-cell type conservation at sequence-driven vs. chromatin-driven sites.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


if __name__ == "__main__":
    X = np.array(([('chr1', 12000,12), ('chr1',50000,12), ('chr1', 10000000,12), ('chr1',130694050, 130694050 )]))
    print X
    np.savetxt('/Users/asheesh/Desktop/test.txt', X, fmt='%s')
