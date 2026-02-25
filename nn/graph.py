import torch
from collections import defaultdict
from nn.data import group_reviews_by_user


def build_hypergraph_data(reviews):
    business_ids = list({r.business_id for r in reviews})
    business_to_idx = {bid: i for i, bid in enumerate(business_ids)}

    user_to_businesses = group_reviews_by_user(reviews)

    user_to_businesses = {
        u: b_list
        for u, b_list in user_to_businesses.items()
        if len(b_list) >= 3
    }

    node_indices = []
    hyperedge_indices = []

    for hyperedge_id, (user_id, business_list) in enumerate(user_to_businesses.items()):
        for bid in business_list:
            node_indices.append(business_to_idx[bid])
            hyperedge_indices.append(hyperedge_id)

    hyperedge_index = torch.tensor(
        [node_indices, hyperedge_indices],
        dtype=torch.long
    )

    rating_sum = defaultdict(float)
    rating_count = defaultdict(int)

    for r in reviews:
        rating_sum[r.business_id] += r.stars
        rating_count[r.business_id] += 1

    x = torch.zeros(len(business_to_idx), 1)

    for bid, idx in business_to_idx.items():
        x[idx] = rating_sum[bid] / rating_count[bid]

    return x, hyperedge_index, business_to_idx