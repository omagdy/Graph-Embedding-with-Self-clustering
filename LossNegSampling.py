import torch
import torch.nn as nn
import numpy as np
import math
from LossNegSampling import *
import random


final_loss_list=[]


class LossNegSampling(nn.Module):
    
    def __init__(self, num_nodes, emb_dim,nb_labels, sequence_length, context_size, no_of_sequences_per_node):
        
        super(LossNegSampling, self).__init__()

        self.V=num_nodes
        self.dim = emb_dim
        self.t=1
        self.gamma_o = 0.001 # Initial Clustering Weight Rate 0.1, 0.01, 0.001
        self.gamma = 0.001
        self.l = sequence_length
        self.w = context_size
        self.N = no_of_sequences_per_node

        self.embedding_u = nn.Embedding(num_nodes, emb_dim) #  embedding  u
        self.embedding_com = nn.Embedding(nb_labels, emb_dim) #  embedding  community centers
       
        self.logsigmoid = nn.LogSigmoid()
    
        initrange = (2.0 / (num_nodes + emb_dim))**0.5 # Xavier init 2.0/sqrt(num_nodes+emb_dim)
        self.embedding_u.weight.data.uniform_(-initrange, initrange) # init u
            
        self.nb_labels= nb_labels
        self.lr_o = 0.001 # Initial Learning Rate 0.01, 0.005
        self.lr_f = 0.0001 # 0.001 0.0005
        self.lr = 0.001
  
        inits=[]
        for k in range(nb_labels):
            rnd_node=torch.tensor([random.randint(0,num_nodes-1)])
            vec=self.embedding_u(rnd_node).data.cpu().numpy()[0]
            inits.append(vec)

        self.embedding_com.weight.data.copy_(torch.from_numpy(np.array(inits))) ##init_communities
        

    def calculate_node_labels(self):

        u_embed = self.embedding_u.weight.data
        c_embed = self.embedding_com.weight.data

        n = u_embed.shape[0]
        d = u_embed.shape[1]        

        k = c_embed.shape[0]

        z = u_embed.reshape(n,1,d)
        z = z.repeat(1,k,1)   
     
        mu = c_embed.reshape(1,k,d)
        mu = mu.repeat(n,1,1)
        
        dist = (z-mu).norm(2,dim=2).reshape((n,k))

        cluster_choice=torch.argmin(dist,dim=1)

        return cluster_choice

    def calculate_new_cluster_centers(self, node_labels):
        nodes_per_cluster = {}
        for i in range(self.nb_labels):
            nodes_per_cluster[i]=[]

        for i in range(len(node_labels)):
            a = int(node_labels[i])
            nodes_per_cluster[a].append(self.embedding_u.weight.data[i])

        for i in range(self.nb_labels):
            npc = nodes_per_cluster[i]
            w=0
            for j in range(len(npc)):
                w+=npc[j]
            w=w/len(npc)
            if type(w)!=int:
                self.embedding_com.weight.data[i]=w

    # Calculate
    def sq_loss_clusters(self, nodes_emb, centroids):
        return ((nodes_emb[:, None]-centroids[None])**2).sum(2).min(1)[0].mean()
        
    def forward(self, u_node, v_node, negative_nodes):
        

        u_embed = self.embedding_u(u_node) # B x 1 x Dim  edge (u,v)
        v_embed = self.embedding_u(v_node) # B x 1 x Dim  
                           
        negs = -self.embedding_u(negative_nodes) # B x K x Dim  neg samples
     
        positive_score=  v_embed.bmm(u_embed.transpose(1, 2)).squeeze(2) # Bx1
        negative_score= torch.sum(negs.bmm(u_embed.transpose(1, 2)).squeeze(2), 1).view(negative_nodes.size(0), -1) # BxK -> Bx1
             
        sum_all = self.logsigmoid(positive_score)+ self.logsigmoid(negative_score)
            
        loss= -torch.mean(sum_all)


        c_embed = self.embedding_com.weight.data

        n = u_embed.shape[0]
        d = u_embed.shape[2]

        node_emb = u_embed.reshape(n,d)

        loss2 = self.sq_loss_clusters(node_emb, c_embed)

        #n = u_embed.shape[0]
        #d = u_embed.shape[2]
        #k = c_embed.shape[0]
        #z = u_embed.reshape(n,1,d)
        #z = z.repeat(1,k,1)   
        #mu = c_embed.reshape(1,k,d)
        #mu = mu.repeat(n,1,1)
        #dist = (z-mu).norm(2,dim=2).reshape((n,k))
        #loss2= self.logsigmoid((dist.min(dim=1)[0]**2)).mean()
        # cluster_choice=torch.argmin(dist,dim=1)

        final_loss=loss+(loss2*self.gamma)

        # return final_loss, cluster_choice
        return final_loss

    
    def get_emb(self, input_node):
        
        embeds = self.embedding_u(input_node) ### u

        return embeds
