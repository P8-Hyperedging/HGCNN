import numpy as np
from collections import defaultdict
from data.data import group_reviews_by_user, Business
from data.data import OpeningHours
import torch
from utils.qualityutils import *
#from word2vecy import train_word2vec, business_to_vec

MAX_POSSIBLE_VARIANCE = 4.0 # ((1-3)^2+(5-3)^2)/2

def build_hypergraph_incidence_matrix(reviews):
    """ Builds a hypergraph incidence matrix H. Rows/nodes are businesses, columns/hyperedges are users. 
    H[i, j] = 1 if business i was reviewed by user j. """

    business_ids, business_to_idx, user_to_businesses = get_business_id_mapping(reviews)

    # matrix size
    num_nodes = len(business_ids)
    num_hyperedges = len(user_to_businesses)

    # initialize incidence matrix
    H = np.zeros((num_nodes, num_hyperedges), dtype=float)

    # fill incidence matrix
    user_column = 0
    for user_id in user_to_businesses:
        reviewed_businesses = user_to_businesses[user_id]
        for business_id, stars in reviewed_businesses:
            business_row = business_to_idx[business_id]
            H[business_row, user_column] = 1
        user_column += 1

    return H, business_ids, business_to_idx

def get_business_id_mapping(reviews):
    # group businesses by user
    user_to_businesses = group_reviews_by_user(reviews)

    # collect all unique business IDs
    business_ids_set = set()
    for businesses in user_to_businesses.values():
       for business_id, stars in businesses:
            business_ids_set.add(business_id)

    business_ids = sorted(business_ids_set)

    # assign each business a row index
    business_to_idx = {}
    for i in range(len(business_ids)):
        business_to_idx[business_ids[i]] = i

    return business_ids, business_to_idx, user_to_businesses

def get_opening_hours_vector(business_hours : OpeningHours):
    '''Extracts opening hours information from a business and encodes it as a vector.'''
    opening_hours = business_hours.hours
    oh_features = np.full(14, -1.0)  # We use -1 for missing data, since 0 represents midnight

    for i in range(len(opening_hours)):
        if opening_hours[i] is None:
            continue
        datetime = opening_hours[i] 
        min_since_midnight = datetime.hour * 60 + datetime.minute
        oh_features[i] = min_since_midnight

    return oh_features

def reviews_from_user(user_id, reviews):
    result = []
    for r in reviews:
        if r['user_id'] == user_id:
            result.append(r)
    return result

# Matrix is N X E, where N is number of nodes (businesses) and E is number of hyperedges (users).
def create_quality_matrix_from_H(reviews):
    business_ids, business_to_idx, user_to_businesses = get_business_id_mapping(reviews)
    
    user_reviews_map = {}
    for review in reviews:
        user_id = review['user_id']
        if user_id not in user_reviews_map:
            user_reviews_map[user_id] = []
        user_reviews_map[user_id].append(review)

    num_hyperedges = len(user_to_businesses)
    W = np.zeros(num_hyperedges, dtype=float) # array which represents diagonal matrix.

    for user_column, user_id in enumerate(user_to_businesses):
        if user_column % 1000 == 0:
            print(f"Calculating quality scores for each hyperedge... ({user_column+1}/{num_hyperedges})")
        
        user_reviews = user_reviews_map[user_id]
        mean = calculate_mean_stars(user_reviews)
        variance = calculate_review_variance(user_reviews, mean)

        W[user_column] = 1 - variance / MAX_POSSIBLE_VARIANCE
    
    return W 


def create_business_feature_matrix(businesses: list[Business], opening_hours):
    nodes = len(businesses)
    num_categories = len(businesses[0].categories)
    total_features = 17 + num_categories
    fm = np.zeros((nodes, total_features))
    print(f"Number of businesses: {len(businesses)}")
    print(f"Number of opening_hours: {len(opening_hours)}")

    for i in range(nodes):
        #vec = business_to_vec(businesses[i].name)
        b = businesses[i]
        oh = opening_hours[i]
        fm[i, 0] = b.review_count
        fm[i, 1] = b.longitude
        fm[i, 2] = b.latitude 
        fm[i, 3:17] = get_opening_hours_vector(oh)
        #fm[i, 17:17+len(vec)] = vec
    return fm

# All nodes need to have SOME label for the KNN pre-processing from the paper
# In practice, we can make some of them -1 and just not use them for training/testing.
def create_label_vector(businesses):
    labels = np.zeros(len(businesses))
    for i in range(len(businesses)):
        b = businesses[i]
        labels[i] = round(b.stars * 2) - 2 # -2 to make class numbers 0-8 instead of 2-10
        
    return labels

def rand_train_test_idx_simple(n_nodes, train_prop=0.75) -> tuple[torch.Tensor, torch.Tensor]:
    """ Simple random split. Train proportion is provide, the rest is validation. """
    n = n_nodes
    train_num = int(n * train_prop)
    valid_num = int(n * (1-train_prop))
    
    perm = torch.as_tensor(np.random.permutation(n))
    
    train_idx = perm[:train_num]
    valid_idx = perm[train_num:train_num + valid_num]
    
    return train_idx, valid_idx