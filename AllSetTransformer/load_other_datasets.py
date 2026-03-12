'''This whole file is adapted from https://github.com/jianhao2016/AllSet'''

import torch

import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append('../nn/')
from data.data import load_postgres_review_data, group_reviews_by_user

def load_yelp_dataset(
        limit=100000, min_reviews_per_user=5, 
        name_dictionary_size = 1000,
        train_percent = 0.025):
    '''
    Load yelp dataset from database using data.py functionality.
    Then build hypergraph where nodes are business and hyperedges are user reviews.
    '''
    # Utilize data.py to load review data
    reviews = load_postgres_review_data(limit=limit, min_reviews_per_user=min_reviews_per_user)

    business_dict = {}
    for review in reviews:
        business_dict[review.business.business_id] = review.business
    businesses = list(business_dict.values())

    business_to_id = {b.business_id: id for id, b in enumerate(businesses)}
    num_nodes = len(businesses)

    feature_list = []
    for b in businesses:
        feature_list.append([
            b.latitude, 
            b.longitude,
            b.stars,
            b.review_count])
        
    features = np.array(feature_list)


    labels = np.array([b.stars for b in businesses])

    user_to_business = group_reviews_by_user(reviews)

    node_list = []
    edge_list = []
    for hyperedge_id, (user_id, buseness_ids) in enumerate(user_to_business.items()):
        for business_id in buseness_ids:
            if business_id in business_to_id:
                node_id = business_to_id[business_id]
                node_list.append(node_id)
                edge_list.append(hyperedge_id + num_nodes)

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features, edge_index=edge_index, y=labels)
                
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
            data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.n_x = num_nodes
    data.train_percent = train_percent
    data.num_hyperedges = len(user_to_business)
    return data

#     # first load node features:
#     # load longtitude and latitude of restaurant.
#     latlong = pd.read_csv(osp.join(path, 'yelp_restaurant_latlong.csv')).values

#     # city - zipcode - state integer indicator dataframe.
#     loc = pd.read_csv(osp.join(path, 'yelp_restaurant_locations.csv'))
#     state_int = loc.state_int.values
#     city_int = loc.city_int.values

#     num_nodes = loc.shape[0]
#     state_1hot = np.zeros((num_nodes, state_int.max()))
#     state_1hot[np.arange(num_nodes), state_int - 1] = 1

#     city_1hot = np.zeros((num_nodes, city_int.max()))
#     city_1hot[np.arange(num_nodes), city_int - 1] = 1

#     # convert restaurant name into bag-of-words feature.
#     vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
#     res_name = pd.read_csv(osp.join(path, 'yelp_restaurant_name.csv')).values.flatten()
#     name_bow = vectorizer.fit_transform(res_name).todense()

#     features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

#     # then load node labels:
#     df_labels = pd.read_csv(osp.join(path, 'yelp_restaurant_business_stars.csv'))
#     labels = df_labels.values.flatten()

#     num_nodes, feature_dim = features.shape
#     assert num_nodes == len(labels)
#     print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

#     features = torch.FloatTensor(features)
#     labels = torch.LongTensor(labels)

#     # The last, load hypergraph.
#     # Yelp restaurant review hypergraph is store in a incidence matrix.
#     H = pd.read_csv(osp.join(path, 'yelp_restaurant_incidence_H.csv'))
#     node_list = H.node.values - 1
#     edge_list = H.he.values - 1 + num_nodes

#     edge_index = np.vstack([node_list, edge_list])
#     edge_index = np.hstack([edge_index, edge_index[::-1, :]])

#     edge_index = torch.LongTensor(edge_index)

#     data = Data(x = features,
#                 edge_index = edge_index,
#                 y = labels)

#     # data.coalesce()
#     # There might be errors if edge_index.max() != num_nodes.
#     # used user function to override the default function.
#     # the following will also sort the edge_index and remove duplicates. 
#     total_num_node_id_he_id = edge_index.max() + 1
#     data.edge_index, data.edge_attr = coalesce(data.edge_index, 
#             None, 
#             total_num_node_id_he_id, 
#             total_num_node_id_he_id)
            

#     n_x = num_nodes
# #     n_x = n_expanded
#     num_class = len(np.unique(labels.numpy()))
#     val_lb = int(n_x * train_percent)
#     percls_trn = int(round(train_percent * n_x / num_class))
#     # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
#     data.n_x = n_x
#     # add parameters to attribute
    
#     data.train_percent = train_percent
#     data.num_hyperedges = H.he.values.max()
    
#     return data