import os

import torch
from torch_geometric.data import Data

from nn.data import load_business_data, load_review_data, sort_businesses_by_review_count, load_user_data
from nn.graph import build_bipartite_graph, visualize_bipartite

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

base_path = os.path.dirname(os.path.realpath(__file__))

businesses = load_business_data(base_path)
users = load_user_data(base_path)
reviews = load_review_data(businesses, users, base_path, limit=50)

print("Sample Reviews:")
for r in reviews[:5]:
    print(r)

G = build_bipartite_graph(reviews)
visualize_bipartite(G)


# Plan: Figure out what to predict.
# - Let's do star ratings, then the recommender system can recommend if stars are high enough.

# - How do we construct the graph to make this prediction?
# - A few things worth trying:
# For regular GNNs:
#   - Just the bipartite graph with user and business nodes, and edges representing reviews. 
#    Features could be the star rating of the review, or other metadata about the user/business.
#    Not sure how to make this "mobile"
#   - A unipartite graph where the nodes are reviews, and edges connect reviews that share a user or business.
#   - a unipartite graph where the nodes are businesses, and edges connect businesses that share a user who reviewed both. (PENN Lecture 8.2)
#   - a unipartite graph where the nodes are businesses, and edges connect businesses in the same area...???

# a unipartite graph where the nodes are businesses, and edges connect businesses that share a user who reviewed both. (PENN Lecture 8.2)
# For this, we need to create a similarity matrix between businesses based on shared users. We can use the review data to count how many users reviewed both businesses, and use that as a weight for the edge between those two business nodes. 
# The features for each business node could be the average star rating, or other metadata about the business.
