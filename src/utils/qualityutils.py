import numpy as np
import torch

def calculate_mean_stars(reviews):
    mean_stars = 0
    for review in reviews:
        mean_stars += review['stars']
    mean_stars /= len(reviews)

    return mean_stars

def calculate_review_variance(reviews, mean_stars):
    variance = 0
    for review in reviews:
        variance += (review['stars'] - mean_stars) ** 2
    variance /= len(reviews)

    return variance

def calculate_centroid(feature_matrix):
    return np.mean(feature_matrix, axis=0)

def calculate_total_distance_to_centroid(local_feature_matrix, centroid):
    distances = np.linalg.norm(local_feature_matrix - centroid, axis=1)
    return np.sum(distances)

def calculate_centroid_torch(feature_matrix):
    return torch.mean(feature_matrix, dim=0)

def calculate_total_distance_to_centroid_torch(local_feature_matrix, centroid):
    distances = torch.norm(local_feature_matrix - centroid, dim=1)
    return torch.sum(distances)