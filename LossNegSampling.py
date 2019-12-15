import torch
import torch.nn as nn
from LossNegSampling import *
import numpy as np

class LossNegSampling(nn.Module):
    
    def __init__(self, num_nodes, emb_dim):
 
        super(LossNegSampling, self).__init__()
        self.embedding_u = nn.Embedding(num_nodes, emb_dim) #  embedding  u
        self.logsigmoid = nn.LogSigmoid()
        self.embedding_com = nn.Embedding(nb_labels, emb_dim)
        initrange = (2.0 / (num_nodes + emb_dim))**0.5 # Xavier init 2.0/sqrt(num_nodes+emb_dim)
        self.embedding_u.weight.data.uniform_(-initrange, initrange) # init u
        self.nb_labels= nb_labels

        self.embedding_com.weight.data.copy_(self.embedding_u(torch.LongTensor(np.random.randint(0,num_nodes,size= nb_labels))))

        
        for i in range(1,5):
            print(i, self.embedding_com.weight[i])

    def forward(self, u_node, v_node, negative_nodes):
            
        u_embed = self.embedding_u(u_node) # B x 1 x Dim  edge (u,v)
        v_embed = self.embedding_u(v_node) # B x 1 x Dim  

    
        negs = -self.embedding_u(negative_nodes) # B x K x Dim  neg samples
     
        positive_score=  v_embed.bmm(u_embed.transpose(1, 2)).squeeze(2) # Bx1
        negative_score= torch.sum(negs.bmm(u_embed.transpose(1, 2)).squeeze(2), 1).view(negative_nodes.size(0), -1) # BxK -> Bx1
             
        sum_all = self.logsigmoid(positive_score)+ self.logsigmoid(negative_score)
            
        loss1= -torch.mean(sum_all)
        
        
        n = u_embed.shape[0]
        d = u_embed.shape[1]
        z = u_embed.repeat(1,self.nb_labels,1)  
        
        mu = self.embedding_com.weight.repeat(n,1,1)

        dist = (z-mu).norm(2,dim=2).reshape((n,self.nb_labels))
        Clusteringcost= (dist.min(dim=1)[0]**2).mean()

        total_loss=loss1+Clusteringcost
    
        return total_loss

    
    def get_emb(self, input_node):
        
        embeds = self.embedding_u(input_node) ### u

        return embeds