import torch
import torch.nn as nn
import numpy as np
from LossNegSampling import *
import random


final_loss_list=[]
gamma0=0.1
class LossNegSampling(nn.Module):
    
    def __init__(self, num_nodes, emb_dim,nb_labels):
        
        super(LossNegSampling, self).__init__()
        V=num_nodes
        L=emb_dim
        t=0
        self.t=t
        self.embedding_u = nn.Embedding(num_nodes, emb_dim) #  embedding  u
        self.embedding_com = nn.Embedding(nb_labels, emb_dim) #  embedding  community centers
       
        self.logsigmoid = nn.LogSigmoid()
    
        initrange = (2.0 / (num_nodes + emb_dim))**0.5 # Xavier init 2.0/sqrt(num_nodes+emb_dim)
        self.embedding_u.weight.data.uniform_(-initrange, initrange) # init u
            
        self.nb_labels= nb_labels

        
        inits=[]
        for k in range(nb_labels):
            rnd_node=torch.tensor([random.randint(0,num_nodes-1)])
            vec=self.embedding_u(rnd_node).data.cpu().numpy()[0]
            inits.append(vec)

        self.embedding_com.weight.data.copy_(torch.from_numpy(np.array(inits))) ##init_communities
        
        #for i in range(1,5):
        #    print(i, self.embedding_com.weight[i])
        
    def forward(self, u_node, v_node, negative_nodes,nb_labels):
        self.t=self.t+1
        u_embed = self.embedding_u(u_node) # B x 1 x Dim  edge (u,v)
        v_embed = self.embedding_u(v_node) # B x 1 x Dim  
           

                
        negs = -self.embedding_u(negative_nodes) # B x K x Dim  neg samples
     
        positive_score=  v_embed.bmm(u_embed.transpose(1, 2)).squeeze(2) # Bx1
        negative_score= torch.sum(negs.bmm(u_embed.transpose(1, 2)).squeeze(2), 1).view(negative_nodes.size(0), -1) # BxK -> Bx1
             
        sum_all = self.logsigmoid(positive_score)+ self.logsigmoid(negative_score)
            
        loss= -torch.mean(sum_all)

        
        
        #gamma=gamma0*(10**(-t*numpy.log10(gamma0)/V*L)
        
        n = u_embed.shape[0]
        d = u_embed.shape[1]        
        z = u_embed.repeat(1,self.nb_labels,1)  
        
     
        #print(u_embed)
        #print(u_embed.size())
        
        mu = self.embedding_com.weight.repeat(n,1,1)
        
        dist = (z-mu).norm(2,dim=2).reshape((n,self.nb_labels))

        loss2= (dist.min(dim=1)[0]**2).mean()

        cluster_choice=torch.argmin(dist,dim=1)
        final_loss=loss+loss2

        return final_loss,cluster_choice

    
    def get_emb(self, input_node):
        
        embeds = self.embedding_u(input_node) ### u

        return embeds
