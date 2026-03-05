import numpy as np

def create_business_feature_matrix(businesses):
    nodes = len(businesses)
    fm = np.zeros((nodes, 3)) # Magic number = number of features per node
    
    for i in range(nodes):
        b = businesses[i]
        fm[i, 0] = b.review_count
        fm[i, 1] = b.longitude
        fm[i, 2] = b.latitude 
        
    return fm

# All nodes need to have SOME label for the KNN pre-processing from the paper
# In practice, we can make some of them -1 and just not use them for training/testing.
def create_label_vector(businesses):
    labels = np.zeros(len(businesses))
    for i in range(len(businesses)):
        b = businesses[i]
        if b.stars >= 4.0:
            labels[i] = 2
        elif b.stars >= 3.0:
            labels[i] = 1
        else:
            labels[i] = 0
        
    return labels