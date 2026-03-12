import numpy as np
from collections import defaultdict
from data.data import group_reviews_by_user
from data.data import OpeningHours

def build_hypergraph_incidence_matrix(reviews):
    '''Builds a hypergraph incidence matrix `H` where nodes are businesses 
       and hyperedges are user reviews. '''
    business_ids = list({r.business.business_id for r in reviews})
    business_to_idx = {bid: i for i, bid in enumerate(business_ids)}

    user_to_businesses = group_reviews_by_user(reviews)

    node_indices = []
    hyperedge_indices = []

    for hyperedge_id, (user_id, business_list) in enumerate(user_to_businesses.items()):
        for bid in business_list:
            node_indices.append(business_to_idx[bid])
            hyperedge_indices.append(hyperedge_id)

    num_nodes = len(business_to_idx)
    num_hyperedges = len(user_to_businesses)

    H = np.zeros((num_nodes, num_hyperedges), dtype=float)

    for node, edge in zip(node_indices, hyperedge_indices):
        H[node, edge] = 1

    return H, business_ids, business_to_idx

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

    if np.all(oh_features == -1):
        print(f"Warning: Business {business_hours.business_id} has no opening hours data.")
    return oh_features

def create_business_feature_matrix(businesses, opening_hours):
    nodes = len(businesses)
    fm = np.zeros((nodes, 17)) # Magic number = number of features per node
    print(f"Number of businesses: {len(businesses)}")
    print(f"Number of opening_hours: {len(opening_hours)}")
    
    for i in range(nodes):
        b = businesses[i]
        oh = opening_hours[i]
        
        fm[i, 0] = b.review_count
        fm[i, 1] = b.longitude
        fm[i, 2] = b.latitude 
        fm[i, 3:17] = get_opening_hours_vector(oh)
    return fm

# All nodes need to have SOME label for the KNN pre-processing from the paper
# In practice, we can make some of them -1 and just not use them for training/testing.
def create_label_vector(businesses):
    labels = np.zeros(len(businesses))
    for i in range(len(businesses)):
        b = businesses[i]
        if b.stars >= 4.0:
            labels[i] = 2
        elif b.stars >= 3.0:
            labels[i] = 1
        else:
            labels[i] = 0
        
    return labels