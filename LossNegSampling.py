import torch
import torch.nn as nn
from LossNegSampling import *

class LossNegSampling(nn.Module):

    def __init__(self, num_nodes, emb_dim):
        super(LossNegSampling, self).__init__()

        self.embedding_u = nn.Embedding(num_nodes, emb_dim)  # embedding  u
        self.logsigmoid = nn.LogSigmoid()

        initrange = (2.0 / (num_nodes + emb_dim)) ** 0.5  # Xavier init 2.0/sqrt(num_nodes+emb_dim)
        self.embedding_u.weight.data.uniform_(-initrange, initrange)  # init u

    def forward(self, u_node, v_node, negative_nodes):
        u_embed = self.embedding_u(u_node)  # B x 1 x Dim  edge (u,v)
        v_embed = self.embedding_u(v_node)  # B x 1 x Dim

        negs = -self.embedding_u(negative_nodes)  # B x K x Dim  neg samples

        positive_score = v_embed.bmm(u_embed.transpose(1, 2)).squeeze(2)  # Bx1
        negative_score = torch.sum(negs.bmm(u_embed.transpose(1, 2)).squeeze(2), 1).view(negative_nodes.size(0),
                                                                                         -1)  # BxK -> Bx1

        sum_all = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)

        loss = -torch.mean(sum_all)

        return loss

    def get_emb(self, input_node):
        embeds = self.embedding_u(input_node)  ### u

        return embeds