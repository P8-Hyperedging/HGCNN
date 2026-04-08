'''This whole file is adapted from https://github.com/jianhao2016/AllSet'''

import torch
import numpy as np

from torch_geometric.data import Data
from torch_sparse import coalesce
from collections import Counter
from torch_scatter import scatter_add
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.feature_extraction.text import CountVectorizer

from data.data import load_postgres_review_data, group_reviews_by_user, load_postgres_business_list_data

def load_yelp_dataset(train_percent = 0.025):
    '''
    Load yelp dataset from database using data.py functionality.
    Then build hypergraph where nodes are business and hyperedges are user reviews.
    '''
    # Utilize data.py to load review data
    reviews = load_postgres_review_data()

    business_ids = set(r['business_id'] for r in reviews)
    businesses = load_postgres_business_list_data(list(business_ids))

    business_to_id = {b.business_id: id for id, b in enumerate(businesses)}
    num_nodes = len(businesses)

    # Bag of words for bussiness names, same procedure as in AllSet, but adapted to database fethcing
    vectorizer = CountVectorizer(max_features = 1000, stop_words = 'english', strip_accents = 'ascii')
    res_name = [b.name for b in businesses]
    name_bow = vectorizer.fit_transform(res_name).toarray()

    feature_list = []
    for i, b in enumerate(businesses):
        feature_list.append([
            b.latitude, 
            b.longitude,
            b.review_count,
            *name_bow[i].tolist()])
        
    features = np.array(feature_list)


    labels = np.array([b.stars for b in businesses])

    user_to_business = group_reviews_by_user(reviews)

    node_list = []
    edge_list = []
    for hyperedge_id, (user_id, buseness_ids) in enumerate(user_to_business.items()):
        for business_id, stars in buseness_ids:
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

def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges

    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data


def norm_contruction(data, option='all_one', TYPE='V2E'):
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
            V_norm = Vdeg**(-1/2)
            E_norm = HEdeg**(-1/2)
            data.norm = V_norm[data.edge_index[0]] * \
                E_norm[data.edge_index[1]-cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
    return data


def expand_edge_index(data, edge_th=0):
    '''
    args:
        num_nodes: regular nodes. i.e. x.shape[0]
        num_edges: number of hyperedges. not the star expansion edges.

    this function will expand each n2he relations, [[n_1, n_2, n_3], 
                                                    [e_7, e_7, e_7]]
    to :
        [[n_1,   n_1,   n_2,   n_2,   n_3,   n_3],
         [e_7_2, e_7_3, e_7_1, e_7_3, e_7_1, e_7_2]]

    and each he2n relations:   [[e_7, e_7, e_7],
                                [n_1, n_2, n_3]]
    to :
        [[e_7_1, e_7_2, e_7_3],
         [n_1,   n_2,   n_3]]

    and repeated for every hyperedge.
    '''
    edge_index = data.edge_index
    num_nodes = data.n_x.item()
    if hasattr(data, 'totedges'):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges

    expanded_n2he_index = []

    # start edge_id from the largest node_id + 1.
    cur_he_id = num_nodes
    # keep an mapping of new_edge_id to original edge_id for edge_size query.
    new_edge_id_2_original_edge_id = {}

    # do the expansion for all annotated he_id in the original edge_index
#     ipdb.set_trace()
    for he_idx in range(num_nodes, num_edges + num_nodes):
        # find all nodes within the same hyperedge.
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

#         Trim a hyperedge if its size>edge_th
        if edge_th > 0:
            if size_of_he > edge_th:
                continue

        if size_of_he == 1:
            # there is only one node in this hyperedge -> self-loop node. add to graph.
            #             n2he_with_same_heid.append(selected_he)

            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)

            # ====
#             new_he2n_same_heid = torch.flip(selected_he, dims = [0])
#             he2n_with_same_heid.append(new_he2n_same_heid)

#             new_he2n = torch.flip(selected_he, dims = [0])
#             new_he2n[0] = cur_he_id
#             expanded_he2n_index.append(new_he2n)

            cur_he_id += 1
            continue

        # -------------------------------
#         # new_n2he_same_heid uses same he id for all nodes.
#         new_n2he_same_heid = selected_he.repeat_interleave(size_of_he - 1, dim = 1)
#         n2he_with_same_heid.append(new_n2he_same_heid)

        # for new_n2he mapping. connect the nodes to all repeated he first.
        # then remove those connection that corresponding to the node itself.
        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # new_edge_ids start from the he_id from previous iteration (cur_he_id).
        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # build a mapping between node and it's corresponding edge.
        # e.g. {n_1: e_7_1, n_2: e_7_2}
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
            cur_he_id += 1

        # create n2he by deleting the self-product edge.
        new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id, tmp_edge_id = new_n2he[0, col_idx].item(
            ), new_n2he[1, col_idx].item()
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)


    new_edge_index = torch.cat(expanded_n2he_index, dim=1)
    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx
