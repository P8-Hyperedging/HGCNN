import os
import sys

import numpy as np
import pytest

from utils.qualityutils import calculate_mean_stars, calculate_review_variance, calculate_centroid, calculate_total_distance_to_centroid


def test_calculate_total_distance_to_centroid():
    local_feature_matrix = np.array([
        [1.0, 2.0], # sqrt(-2^2 + -2^2) = 2.8284271247461903
        [3.0, 4.0], # sqrt(0^2 + 0^2) = 0.0
        [5.0, 6.0]  # sqrt(2^2 + 2^2) = 2.8284271247461903
    ])
    centroid = calculate_centroid(local_feature_matrix)
    total_distance = calculate_total_distance_to_centroid(local_feature_matrix, centroid)

    print(f"Centroid: {centroid}")
    print(f"Total distance to centroid: {total_distance}")

    expected_centroid = np.array([3.0, 4.0])
    expected_total_distance = np.linalg.norm(local_feature_matrix - expected_centroid, axis=1).sum()

    assert np.allclose(centroid, expected_centroid)
    assert np.isclose(total_distance, expected_total_distance)