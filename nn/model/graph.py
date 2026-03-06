import torch
from collections import defaultdict
from data.data import group_reviews_by_user

def build_hypergraph_data(reviews, min_reviews_per_user=0):
    business_ids = list({r.business_id for r in reviews})
    business_to_idx = {bid: i for i, bid in enumerate(business_ids)}

    user_to_businesses = group_reviews_by_user(reviews)

    user_to_businesses = {
        u: b_list
        for u, b_list in user_to_businesses.items()
        if len(b_list) >= min_reviews_per_user
    }

    node_indices = []
    hyperedge_indices = []

    for hyperedge_id, (user_id, business_list) in enumerate(user_to_businesses.items()):
        for bid in business_list:
            node_indices.append(business_to_idx[bid])
            hyperedge_indices.append(hyperedge_id)

    # Number of nodes and hyperedges
    num_nodes = len(business_to_idx)
    num_hyperedges = len(user_to_businesses)

    # Create a dense incidence matrix
    H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float)

    # Populate the incidence matrix
    for node, edge in zip(node_indices, hyperedge_indices):
        H[node, edge] = 1

    return H