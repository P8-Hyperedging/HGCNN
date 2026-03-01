import networkx as nx
import matplotlib.pyplot as plt

def build_bipartite_graph(reviews):
    G = nx.Graph()
    for review in reviews:
        G.add_node(review.user.user_id, label=review.user.name, type='user')
        G.add_node(review.business.business_id, label=review.business.name, type='business')
        G.add_edge(review.user.user_id, review.business.business_id)
    return G

def visualize_bipartite(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)  # Layout


    node_colors = []
    for n in G.nodes(data=True):
        if n[1]['type'] == 'user':
            node_colors.append('skyblue')
        else:
            node_colors.append('lightgreen')

    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=500, alpha=0.8, edge_color='gray')

    labels = {n[0]: n[1]['label'] for n in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("User ↔ Business Hypergraph (Bipartite Graph)")
    plt.axis('off')
    plt.show()