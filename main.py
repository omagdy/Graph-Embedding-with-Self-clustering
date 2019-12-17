
#from eval import node_classification, plot_embeddings, link_prediction

import pickle

from lineEmb import *
import os
import numpy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
    
if __name__=='__main__':
    
    for name in [ 'airport']:

        edge_file= '%s.txt'%name
        #label_file= './%s/%s-label.txt'%(name, name)
        
        
        f_social= open(edge_file, 'r')
        
        nb_labels=5
        social_edges=[]
        
        for line in f_social:
            
            a=line.strip('\n').split(' ')
            social_edges.append((a[0],a[1]))

        
        for size in [8]: #50, 100, 200

            model= lineEmb( edge_file,  social_edges, name,  emb_size= size, alpha=5, epoch=6, batch_size=128, shuffel=True)
        
            embeddings= model.train(nb_labels)
        
        print('\n')
        
        for i in range(1,5):
            print(i, embeddings[i])

            #node_classification( embeddings, label_file, name, size)
            #link_prediction(edge_file,  embeddings, name, size)
            #plot_embeddings( embeddings, label_file, name)
            
    print(final_loss_list)
    print(len(final_loss_list))
    
    plt.plot(final_loss_list)
    plt.ylabel('loss')
    plt.show()
