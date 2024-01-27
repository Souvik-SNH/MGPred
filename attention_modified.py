import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
'''
Any node does not need to aggregate all their neighbors to describe the n-th view. Therefore,
we assign different weights to the neighbor nodes by calculating the correlations between neighbor nodes and this node, so as
to obtain neighbor node information differently. For each view, the node attention module first calculates different weights for
each node and its neighbors correspondingly.

The attention mechanism calculates scores to determine the importance of different parts of the input. 
These scores are used to compute a weighted sum of the values, where the weights are determined by the attention scores.
1. The attention scores are often computed by measuring the similarity (dot product, cosine similarity, etc.) between the query and key.
2. The softmax function is applied to the attention scores to obtain a distribution of weights. 
This distribution reflects the importance or relevance of different parts of the input. Higher attention scores lead to higher weights.
3. The weighted sum of the values (associated with the input) is computed using the obtained weights. This weighted sum represents the attention mechanism's output.

Attention mechanisms have significantly improved the performance of models in various tasks by allowing them to selectively focus on different parts of the input, 
capturing dependencies and relationships more effectively.
'''
'''
Example usage:
attention_model = attention(embedding_dim=64, droprate=0.5)

# Create some dummy input tensors (replace these with your actual data)
feature1 = torch.randn(64)  # Example input feature 1
feature2 = torch.randn(64)  # Example input feature 2
n_neighs = 5  # Example number of neighbors

# Apply attention
attended_values = attention_model(feature1, feature2, n_neighs)
'''

class attention(nn.Module):
    def __init__(self, embedding_dim, droprate, cuda = "cpu"):
        super(attention, self).__init__()

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 1)
        self.softmax = torch.nn.functional.softmax()

    # a = np.array([[1,2,3]])
    # a = a.repeat(5, 1)
    # [[1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]]
    # 论文中公式（4）中atm在哪？ 就是第一层隐藏层中的待训练的w把？
    '''
    Given two input features (feature1 and feature2) and the number of neighbors (n_neighs), 
    it concatenates feature1 with replicated feature2 for each neighbor. 
    Then, it applies the linear layer att1, followed by a ReLU activation and dropout. The result is passed through att2 to get attention weights.
    '''
    def forward(self, feature1, feature2, n_neighs):
        # feature2 = feature2.detach().numpy()
        # print(feature1.detach().numpy())
        # print(feature2.detach().numpy())
        # print(n_neighs)
        # 先传该节点所有邻居的特征，然后传该节点的特征
        feature2_reps = feature2.repeat(n_neighs, 1)

        x = torch.cat((feature1, feature2_reps), 1)
        x = F.relu(self.att1(x).to(self.device), inplace=True)
        x = F.dropout(x, training=self.training, p=self.droprate)
        # print(x.detach().numpy())
        x = self.att2(x).to(self.device)

        att = torch.nn.functional.softmax(x, dim=0)
        # xxxx = att.detach().numpy() # 返回一个样本的所有邻居的各自权重的值，然后把它标准化
        return att