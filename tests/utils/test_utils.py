# test_eu_dis.py
import os
import sys

import numpy as np
import pytest

from utils.utils import Eu_dis, _generate_G_from_H

# ---------------
# TEST Eu_dis
# ---------------

def test_eu_dis_basic():
    x = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0]
    ])

    dist = Eu_dis(x)

    print("\nDistance matrix:")
    print(dist)

    assert np.isclose(dist[0, 1], 5.0)
    assert np.isclose(dist[1, 2], 5.0)
    assert np.isclose(dist[0, 2], 10.0)

    # Diagonal should be zero
    assert np.allclose(np.diag(dist), 0.0)

    # Matrix should be symmetric
    assert np.allclose(dist, dist.T)

# ---------------
# TEST _generate_G_from_H
# ---------------

def test_output_shape():
    H = np.array([
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=float)
    G = _generate_G_from_H(H)
    assert G.shape == (3, 3)

def test_symmetric():
    H = np.array([
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=float)
    G = _generate_G_from_H(H)
    assert np.allclose(G, G.T)

def test_variable_weight_reconstruction():
    H = np.array([
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=float)

    G_direct = _generate_G_from_H(H, variable_weight=False)
    DV2_H, W, invDE_HT_DV2 = _generate_G_from_H(H, variable_weight=True)
    G_parts = DV2_H @ W @ invDE_HT_DV2

    assert np.allclose(G_direct, G_parts)

def test_single_hyperedge_two_nodes():
    H = np.array([
        [1],
        [1]
    ], dtype=float)

    G = _generate_G_from_H(H)
    expected = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    assert np.allclose(G, expected)

def test_disconnected_groups():
    H = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ], dtype=float)

    G = _generate_G_from_H(H)

    assert np.allclose(G[:2, 2:], 0)
    assert np.allclose(G[2:, :2], 0)