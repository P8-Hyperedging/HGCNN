import numpy as np
import pytest
from data.n_preprocessing import build_hypergraph_incidence_matrix


def create_reviews(user_business_pairs):
    """ Create reviews from list of (user_id, business_id) tuples. """
    reviews = []
    review_id = 0
    
    for user_id, business_id in user_business_pairs:
        reviews.append({
            "review_id": review_id,
            "user_id": user_id,
            "business_id": business_id
        })
        review_id += 1
    
    return reviews


def test_hypergraph_properties():
    """ Testing the structure and properties of the incidence matrix H. """
    # fixed dataset with known connections
    user_business_pairs = [
        ("u1", "b1"),
        ("u1", "b2"),
        ("u2", "b2"),
        ("u2", "b3"),
        ("u3", "b1"),
        ("u3", "b3"),
    ]
    
    reviews = create_reviews(user_business_pairs)
    H, business_ids, business_to_idx = build_hypergraph_incidence_matrix(reviews)
    
    print("\nIncidence matrix:")
    print(H)
    print(f"Shape: {H.shape}")
    
    # sanity checks on the structure of the incidence matrix
    assert H.shape == (3, 3), f"Expected 3 businesses and 3 users, got {H.shape}"
    assert np.all((H == 0) | (H == 1)), "Matrix should only contain 0s and 1s"
    
    # checking specific connections
    u1_idx = 0
    b1_idx = business_to_idx["b1"]
    b2_idx = business_to_idx["b2"]
    
    assert H[b1_idx, u1_idx] == 1, "u1 should be connected to b1"
    assert H[b2_idx, u1_idx] == 1, "u1 should be connected to b2"
    
    # every hyperedge (user) should have at least one node (business)
    assert np.all(np.sum(H, axis=0) > 0), "Each user should review at least one business"