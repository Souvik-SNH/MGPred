import torch
import heapq
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.l0dense import L0Dense
from torch.autograd import Variable
from utils.attention import attention



class aggregator(nn.Module):
    """
    aggregator: for aggregating feature of neighbors
    """
    def __init__(self, drug_feature, side_feature, adj, embed_dim, weight_decay = 0.0005, droprate = 0.5,
                    cuda = "cpu", is_drug_part = True):
        super(aggregator, self).__init__()

        self.drugfeature = drug_feature
        self.sidefeature = side_feature
        self.adj = adj
        self.embed_dim = embed_dim
        self.weight_decay = weight_decay
        self.droprate = droprate
        self.device = cuda
        self.is_drug = is_drug_part

        # L0-regularized linear layers for user and item (drug or side) features.
        self.u_layer = L0Dense(self.drugfeature.embedding_dim, self.embed_dim,
                                weight_decay = self.weight_decay, droprate = self.droprate)
        self.i_layer = L0Dense(self.sidefeature.embedding_dim, self.embed_dim,
                                weight_decay = self.weight_decay, droprate = self.droprate)
        # Attention mechanism for calculating attention weights
        self.att = attention(self.embed_dim, self.droprate, cuda = self.device)
    def forward(self, nodes): # nodes: List of node indices for which features are to be aggregated

        # These lines fetches the features of drugs/Side effect represented by the nodes in the input list,
        # applies a transformation with the user layer, and ensures that the resulting features are on the correct computational device.
        if self.is_drug:
            embed_matrix = torch.empty(self.drugfeature.num_embeddings, self.embed_dim,
                                        dtype = torch.float).to(self.device) # It initializes an empty matrix (embed_matrix) for aggregated embeddings.
            nodes_fea = self.u_layer(self.drugfeature.weight[nodes]).to(self.device) # fetches the node features .
                                                    # The 'nodes' list contains indices of the nodes (drugs) for which we want to obtain features.
                                                    # self.u_layer(...) this applies the user layer (u_layer) to the selected drug features.
                                                    # The u_layer is an instance of L0Dense or a similar layer, and it seems to be responsible for transforming the drug features.
            threshold = 24 # A threshold is set, which seems to be used in deciding whether to apply reservoir sampling
                            # If there are interactions, it checks if the number of interactions is less than the threshold (24).
                            # If so, all interactions are considered. Otherwise, reservoir sampling is applied to select 24 interactions.
        else:
            embed_matrix = torch.empty(self.sidefeature.num_embeddings, self.embed_dim,
                                        dtype = torch.float).to(self.device)
            nodes_fea = self.i_layer(self.sidefeature.weight[nodes]).to(self.device)
            threshold = 24

        length = list(nodes.size())
        for i in range(length[0]): #  It iterates over the nodes provided in the nodes input.
            index = nodes[[i]].cpu().numpy() # fetches the index of the current node.
            if self.training:
                interactions = self.adj[index[0]] # if the model is in training mode (self.training), it directly fetches the interactions for the current node from self.adj.
                                                # These interactions might represent neighboring nodes or items connected to the current node.
            else:
                if index[0] in self.adj.keys(): # If not in training mode, it checks whether the index is present in self.adj.keys().
                    interactions = self.adj[index[0]]
                else: # If not, it calculates the node feature (embedding) for the current node.
                    if self.is_drug: #  it uses the user layer (self.u_layer) to get the feature from self.drugfeature.weight.
                        node_feature = self.u_layer(self.drugfeature.weight[torch.LongTensor(index)]).to(self.device)
                    else: #  it uses the item layer (self.uilayer) to get the feature from self.sidefeature.weight.
                        node_feature = self.i_layer(self.sidefeature.weight[torch.LongTensor(index)]).to(self.device)

                    embed_matrix[index] = node_feature # This newly calculated feature is then stored in the embed_matrix at the current index, and the loop continues to the next iteration
                    continue

            n = len(interactions)
            if n < threshold: # If the number of interactions is less than a predefined threshold (threshold), it directly uses all interactions;
                interactions = [item[0] for item in interactions]
                times = n # number of interactions/neighbour
            else: # otherwise, it subsamples interactions using the a_res function.
                interactions = a_res(interactions, threshold)
                times = threshold
            # Feature retrieval for Subsampled Interactions
            # In the drug case, it gets features from the item layer (self.i_layer), and in the item case, it gets features from the user layer (self.u_layer).
            if self.is_drug:
                neighs_feature = self.i_layer(self.sidefeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.u_layer(self.drugfeature.weight[torch.LongTensor(index)]).to(self.device)
            else:
                neighs_feature = self.u_layer(self.drugfeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.i_layer(self.sidefeature.weight[torch.LongTensor(index)]).to(self.device)
            att_w = self.att(neighs_feature, node_feature, times)
            embedding = torch.mm(neighs_feature.t(), att_w) # performs a matrix multiplication between the transpose of neighs_feature and att_w.
                                                    # This results in a vector representing the weighted sum of features of neighboring nodes/items.
            embed_matrix[index] = embedding.t() # embedding.t() transposes the resulting vector, and this transposed vector is assigned to the embed_matrix at the current index.
                                        # The embed_matrix is a tensor initialized earlier to store the embeddings of all nodes
        return nodes_fea, embed_matrix

def a_res(samples, m):
    """
    samples: [(entity, weight), ...]
    m: number of selected entities
    returns: [(entity, weight), ...]
    """
    heap = []
    for sample in samples:
        w = sample[1]
        u = random.uniform(0, 1)
        k = u ** (1/w)

        if len(heap) < m:
            heapq.heappush(heap, (k, sample))
        elif k > heap[0][0]:
            heapq.heappush(heap, (k, sample))

            if len(heap) > m:
                heapq.heappop(heap)
    return [item[1][0] for item in heap]
