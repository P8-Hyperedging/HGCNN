# test_eu_dis.py
import os
import sys

import numpy as np
import pytest

# Add project root to path so imports like "from nn.utils" work when running pytest from nn/tests


from utils import Eu_dis

def test_eu_dis_basic():
    # 3 samples, 2 features
    x = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0]
    ])

    dist = Eu_dis(x)

    print("\nDistance matrix:")
    print(dist)

    # Expected distances:
    # Between row 0 and 1 -> 5
    # Between row 1 and 2 -> 5
    # Between row 0 and 2 -> 10

    assert np.isclose(dist[0, 1], 5.0)
    assert np.isclose(dist[1, 2], 5.0)
    assert np.isclose(dist[0, 2], 10.0)

    # Diagonal should be zero
    assert np.allclose(np.diag(dist), 0.0)

    # Matrix should be symmetric
    assert np.allclose(dist, dist.T)