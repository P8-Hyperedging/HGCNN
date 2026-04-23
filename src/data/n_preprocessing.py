import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from data.data import group_reviews_by_user, Business
from data.data import OpeningHours
import torch
from utils.qualityutils import *


@dataclass
class FeatureConfig:
    use_review_count: bool = True
    use_location: bool = True
    use_opening_hours: bool = True
    use_categories: bool = True
    use_is_open: bool = True
    use_state_onehot: bool = True
    use_city_freq: bool = True


def fit_categorical_encoders(businesses: list[Business]) -> dict:
    state_categories = sorted({b.state for b in businesses if b.state is not None})
    city_counts = Counter(b.city for b in businesses if b.city is not None)
    n = len(businesses)
    city_freq = {city: count / n for city, count in city_counts.items()}
    return {"states": state_categories, "city_freq": city_freq}

MAX_POSSIBLE_VARIANCE = 4.0 # ((1-3)^2+(5-3)^2)/2

def build_hypergraph_incidence_matrix(reviews):
    """ Builds a hypergraph incidence matrix H. Rows/nodes are businesses, columns/hyperedges are users.
    H[i, j] = 1 if business i was reviewed by user j. Returns a CSR sparse matrix. """
    from scipy.sparse import lil_matrix, csr_matrix

    business_ids, business_to_idx, user_to_businesses = get_business_id_mapping(reviews)

    num_nodes = len(business_ids)
    num_hyperedges = len(user_to_businesses)

    H = lil_matrix((num_nodes, num_hyperedges), dtype=float)

    for user_column, user_id in enumerate(user_to_businesses):
        for business_id, _ in user_to_businesses[user_id]:
            H[business_to_idx[business_id], user_column] = 1

    return csr_matrix(H), business_ids, business_to_idx

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
        # if user_column % 1000 == 0:
            # print(f"Calculating quality scores for each hyperedge... ({user_column+1}/{num_hyperedges})")
        
        user_reviews = user_reviews_map[user_id]
        mean = calculate_mean_stars(user_reviews)
        variance = calculate_review_variance(user_reviews, mean)

        W[user_column] = 1 - variance / MAX_POSSIBLE_VARIANCE
    
    return W 


def create_business_feature_matrix(businesses: list[Business], opening_hours,
                                    config: FeatureConfig = None, encoders: dict = None):
    if config is None:
        config = FeatureConfig()
    if encoders is None:
        encoders = fit_categorical_encoders(businesses)

    state_categories = encoders["states"]
    city_freq = encoders["city_freq"]
    state_to_idx = {s: i for i, s in enumerate(state_categories)}

    nodes = len(businesses)
    num_categories = len(businesses[0].categories)

    n_review_count = 1 if config.use_review_count else 0
    n_location    = 2 if config.use_location else 0
    n_hours       = 14 if config.use_opening_hours else 0
    n_categories  = num_categories if config.use_categories else 0
    n_is_open     = 1 if config.use_is_open else 0
    n_state       = len(state_categories) if config.use_state_onehot else 0
    n_city        = 1 if config.use_city_freq else 0
    total_features = n_review_count + n_location + n_hours + n_categories + n_is_open + n_state + n_city

    fm = np.zeros((nodes, total_features))
    print(f"Number of businesses: {nodes}, Feature matrix shape: {fm.shape}")
    print(f"  review_count={n_review_count}, location={n_location}, hours={n_hours}, "
          f"categories={n_categories}, is_open={n_is_open}, state_onehot={n_state}, city_freq={n_city}")

    for i, (b, oh) in enumerate(zip(businesses, opening_hours)):
        col = 0
        if config.use_review_count:
            fm[i, col] = b.review_count;        
            col += 1

        if config.use_location:
            fm[i, col] = b.latitude;
            col += 1
            fm[i, col] = b.longitude;        
            col += 1

        if config.use_opening_hours:
            fm[i, col:col+14] = get_opening_hours_vector(oh); 
            col += 14

        if config.use_categories:
            n_cat = len(b.categories)
            fm[i, col:col+n_cat] = b.categories; col += n_cat

        if config.use_is_open:
            fm[i, col] = float(b.is_open or 0)
            col += 1

        if config.use_state_onehot:
            if b.state in state_to_idx:
                fm[i, col + state_to_idx[b.state]] = 1.0
            col += len(state_categories)

        if config.use_city_freq:
            fm[i, col] = city_freq.get(b.city, 0.0)
            col += 1

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