import numpy as np
from collections import defaultdict
from data.data import group_reviews_by_user

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