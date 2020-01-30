import torch
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
import nltk
import time
import networkx as nx

from LossNegSampling import *


USE_CUDA = False # torch.cuda.is_available()
#gpus = [0]
#torch.cuda.set_device(gpus[0])
    
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
        
class lineEmb():
    
    def __init__(self, edge_file, social_edges=None, name='wiki', emb_size= 2,  
                     alpha=5, epoch=5, batch_size= 256, shuffel=True , neg_samples=5,
                      sequence_length=21, context_size=10, no_of_sequences_per_node=18):
    
        self.emb_size = emb_size
        self.shuffel = shuffel
    
        self.neg_samples = neg_samples
        self.batch_size=batch_size
        self.epoch=epoch
        self.alpha= alpha
     
        self.name=name
        self.G = nx.read_edgelist(edge_file)
        self.social_edges= social_edges

        self.sequence_length=sequence_length
        self.context_size=context_size
        self.no_of_sequences_per_node=no_of_sequences_per_node
        
        self.index2word = dict()
        self.word2index = dict()
        self.build_vocab()  
        
        self.emb_file= './emb/%s_size_%d_line.emb'%(self.name, self.emb_size)  
                 


    def getBatch(self, batch_size, train_data):
        
        if self.shuffel==True:
            random.shuffle(train_data)
        
        sindex = 0
        eindex = batch_size
        while eindex < len(train_data):
            batch = train_data[sindex: eindex]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch
        
        if eindex >= len(train_data):
            batch = train_data[sindex:]
            
            yield batch
            

    
    def prepare_node(self, node, word2index):
        return Variable(LongTensor([word2index[str(node)]]))
    
             
    def prepare_sequence(self, seq, word2index):
        idxs = list(map(lambda w: word2index[w], seq))

        return Variable(LongTensor(idxs))
    
    #def prepare_weight(self, weight):
        #return  FloatTensor([weight])
    
   
    
    def build_vocab(self):
        self.social_nodes=[]

        for u,v in self.social_edges:
            self.social_nodes.append(u)
            self.social_nodes.append(v)   
            
        self.all_nodes= list(set(self.social_nodes))
        
        self.word2index = {}
        for vo in self.all_nodes:
            if self.word2index.get(vo) is None:
                self.word2index[str(vo)] = len(self.word2index)

        self.index2word = {v:k for k, v in self.word2index.items()}
               
    def prepare_trainData(self, sequences):    
       
        print('prepare training data ...')
        
        self.train_data = []
       
        # for u, v  in self.social_edges:

        #     for i in range(self.alpha):
        #         self.train_data.append(( u, v))
        #         self.train_data.append(( v, u))

        for sequence in sequences:
            for i in range(self.context_size * 2 + 1):
                if i != self.context_size:
                    self.train_data.append((sequence[self.context_size], sequence[i]))

        u_p = []
        v_p = []
        tr_num=0    
        
        for tr in self.train_data:
            
            #print('sample', tr_num, 'len(train_data)', len(self.train_data))
            u_p.append(self.prepare_node(tr[0], self.word2index).view(1, -1))
            v_p.append(self.prepare_node(tr[1], self.word2index).view(1, -1))
            tr_num+=1
            
        train_samples = list(zip(u_p, v_p))
        
        print(len(train_samples), 'samples are ready ...')
        
        return train_samples
        

    def negative_sampling(self, targets,  k):
            
        batch_size = targets.size(0)
        neg_samples = []
         
        for i in range(batch_size):
             
            nsample = []
            target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
            v_node= self.index2word[target_index]
            
            while len(nsample) < k: # num of sampling
                
                neg = random.choice(self.all_nodes)
                if (neg != v_node): 
                        #print(v_node, neg  ) 
          
                    nsample.append(neg)
                           
                else:   
                         
                    continue
               
            neg_samples.append(self.prepare_sequence(nsample, self.word2index).view(1, -1))
        
        return torch.cat(neg_samples)        


    def random_walk_sample(self, no_of_sequences_per_node, sequence_length):
        walks = []
        half_walk = int(sequence_length/2)
        for node in self.all_nodes:
            for i in range(no_of_sequences_per_node):
                node_sequence = []
                walk_1 = self.capture_sequence(node_sequence, node, half_walk)
                walk_1.reverse()
                node_sequence = []
                walk_2 = self.capture_sequence(node_sequence, node, half_walk)
                total_random_walk = walk_1+[node]+walk_2
                walks.append(total_random_walk)
        # random.shuffle(walks)
        # return walks
        flatten = lambda list: [item for sublist in list for item in sublist]
        windows = flatten([list(nltk.ngrams(c, self.context_size * 2 + 1)) for c in walks])
        # print(len(windows[0]))
        #random.shuffle(windows)
        return windows

    #Recursive Function 
    def capture_sequence(self, walk, node, counter):
        if counter==0:
            return walk
        else:
            counter-=1
            connected_nodes = list(self.G[node])
            random_neighbor_node = random.randint(0,len(connected_nodes)-1)
            walk.append(connected_nodes[random_neighbor_node])
            return self.capture_sequence(walk, connected_nodes[random_neighbor_node], counter)
    

    def train (self,nb_labels):
        
        train_data= self.prepare_trainData(self.random_walk_sample(self.no_of_sequences_per_node, self.sequence_length))
        
        final_losses = []
        
        model = LossNegSampling(len(set(self.all_nodes)), self.emb_size, nb_labels,
         self.sequence_length, self.context_size, self.no_of_sequences_per_node)
        
        if USE_CUDA:
           model = model.cuda()
           
        optimizer = optim.Adam(model.parameters(), lr=model.lr) #Learning Rate changed dynamically 
       
        self.epoches=[]
                
        for epoch in range(self.epoch):
            
            t1=time.time()           

            f = open("gamma_values.txt", "a")
            f.write(str(model.gamma)+"\n")
            f.close()
            f = open("alpha_values.txt", "a")
            f.write(str(model.lr)+"\n")
            f.close() 

            for i,  batch in enumerate(self.getBatch(self.batch_size, train_data)):

                model.t+=1 
                model.gamma=model.gamma*(10**((-model.t*np.log10(model.gamma))/(model.l*((model.w*2)+1)*model.V*model.N)))
                lr_f = 0.001
                model.lr = model.lr - ((model.lr-lr_f)*(model.t/(model.l*((model.w*2) +1)*model.V*model.N))) 

            
                inputs, targets= zip(*batch)
               
                inputs= torch.cat(inputs) # B x 1
                targets=torch.cat( targets) # B x 1
    
                negs = self.negative_sampling(targets , self.neg_samples)
    
                model.zero_grad()
                
                final_loss,cluster_choice  = model(inputs, targets, negs,nb_labels)

                final_loss.backward()
                optimizer.step()

                # The changing of the learning rate
                for param in optimizer.param_groups:
                    param['lr'] = model.lr
            
                final_losses.append(final_loss.data.cpu().numpy())
            

            t2= time.time()
            final_loss_list.append(np.mean(final_losses))
            print(self.name, 'loss: %0.3f '%np.mean(final_losses),'Epoch time: ', '%0.4f'%(t2-t1), 'dimension size:', self.emb_size,' Alpha: ', model.lr,' Gamma: ', model.gamma )
                
            #f_loss.write(str('final loss: %0.3f '%np.mean(final_losses) ) +' samples_num: '+str(len(self.train_data))+ str(' epoch time: %0.3f '%(t2-t1) )+
                         #' emb_size: '+str(self.emb_size)+ ' alpha: '+str(self.alpha))
            #f_loss.write('\n')

        #f_loss.close()

        final_emb={}
        normal_emb={}

        #f=open(self.emb_file, 'w')

        for w in self.all_nodes:

            normal_emb[w]=model.get_emb(self.prepare_node(w, self.word2index))

            #f.write(str(w))
            #for j in normal_emb[w].data.cpu().numpy()[0]:
                #f.write(' '+str(j))
            #f.write('\n')
            
            vec=[float(i) for i in normal_emb[w].data.cpu().numpy()[0]]

            final_emb[int(w)]=vec
            
        #f.close()
        
        #write to file
        # inputs, targets= zip(*train_data)
        # inputs= torch.cat(inputs) # B x 1
        # targets=torch.cat(targets) # B x 1
        
        # negs = self.negative_sampling(targets , self.neg_samples)
        # model.zero_grad()
        # final_loss,cluster_choice  = model(inputs, targets, negs,nb_labels)
        # f=open("cluster.txt", 'w')
        # for j in range(cluster_choice.size()[0]):
        #     f.write(str(int(inputs[j]))+' '+str(int(cluster_choice[j])))
        #     f.write('\n')
        # f.close()
        
        return final_emb
