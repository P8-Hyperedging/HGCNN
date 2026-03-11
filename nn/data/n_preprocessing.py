import numpy as np
from collections import defaultdict
from data.data import group_reviews_by_user

def build_hypergraph_incidence_matrix(reviews):
    """ Builds a hypergraph incidence matrix H. Rows/nodes are businesses, columns/hyperedges are users. 
    H[i, j] = 1 if business i was reviewed by user j. """

    # group businesses by user
    user_to_businesses = group_reviews_by_user(reviews)

    # collect all unique business IDs
    business_ids_set = set()
    for businesses in user_to_businesses.values():
        business_ids_set.update(businesses)

    business_ids = list(business_ids_set)

    # assign each business a row index
    business_to_idx = {}
    for i in range(len(business_ids)):
        business_to_idx[business_ids[i]] = i

    # matrix size
    num_nodes = len(business_ids)
    num_hyperedges = len(user_to_businesses)

    # initialize incidence matrix
    H = np.zeros((num_nodes, num_hyperedges), dtype=float)

    # fill incidence matrix
    user_column = 0
    for user_id in user_to_businesses:
        reviewed_businesses = user_to_businesses[user_id]
        for business_id in reviewed_businesses:
            business_row = business_to_idx[business_id]
            H[business_row, user_column] = 1
        user_column += 1

    return H, business_ids, business_to_idx