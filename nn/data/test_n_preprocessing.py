import numpy as np
import pytest
from data.n_preprocessing import *


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


def test_get_opening_hours_vector():
    """ Test the get_opening_hours_vector function with various cases. """
    from data.data import OpeningHours
    
    hours = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    hours[0] = type('datetime', (object,), {'hour': 9, 'minute': 30})() 
    hours[1] = type('datetime', (object,), {'hour': 10, 'minute': 0})()  
    # Open on Monday at 9:30 and Tuesday at 10:00, closed other days

    business_hours = OpeningHours("b1", hours)

    oh_vector = get_opening_hours_vector(business_hours)
    print(f"Opening hours vector: {oh_vector}")
    
    assert oh_vector[0] == 9*60 + 30, "Monday should be open at 9:30 (570 minutes)"
    assert oh_vector[1] == 10*60, "Tuesday should be open at 10:00 (600 minutes)"
    assert np.all(oh_vector[2:] == -1), "Other days should be -1 for missing data/not open"

def test_create_business_feature_matrix():
    """ Test the create_business_feature_matrix function with synthetic data. """
    from data.data import Business, OpeningHours
    
    businesses = [
        Business("b1", "Business 1", 4.0, 100, -122.0, 37.0),
        Business("b2", "Business 2", 3.5, 50, -122.5, 37.5),
        Business("b3", "Business 3", 5.0, 200, -123.0, 38.0)
    ]

    hours = [
        OpeningHours("b1", np.full(14, None)),
        OpeningHours("b2", np.full(14, None)),
        OpeningHours("b3",  np.full(14, None))
    ]

    hours[0].hours[0] = type('datetime', (object,), {'hour': 9, 'minute': 0})()
    hours[1].hours[2] = type('datetime', (object,), {'hour': 10, 'minute': 0})()
    # b1 opens Monday at 9:00, b2 opens Tuesday at 10:00, b3 has no opening hours data

    fm = create_business_feature_matrix(businesses, hours)
    print(f"Feature matrix:\n{fm}")
    
    assert fm.shape == (3, 17), "Feature matrix should have shape (3 businesses, 17 features)"
    assert fm[0, 0] == 100, "First business review count should be 100"
    assert fm[1, 1] == -122.5, "Second business longitude should be -122.5"
    assert fm[2, 2] == 38.0, "Third business latitude should be 38.0"
    assert fm[0, 3] == 9*60, "First business Monday opening should be 540 minutes"
    assert fm[1, 5] == 10*60, "Second business Tuesday opening should be 600 minutes"
    assert np.all(fm[2, 3:] == -1), "Third business should have -1 for all opening hours features"

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