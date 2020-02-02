#from eval import node_classification, plot_embeddings, link_prediction

import pickle

from sklearn.manifold import TSNE

from lineEmb import *
# from Classifier import *
import os
import numpy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
if __name__=='__main__':


    f = open("gamma_values.txt", "w")
    f.write("Gamma Values:"+"\n")
    f.close()

    f = open("alpha_values.txt", "w")
    f.write("Alpha Values:"+"\n")
    f.close()
    
    for name in ['cora-edgelist']:

        edge_file= '%s.txt'%name
        #label_file= './%s/%s-label.txt'%(name, name)
        
        
        f_social= open(edge_file, 'r')
        
        nb_labels = 7 #5
        social_edges=[]
        
        for line in f_social:
            
            a=line.strip('\n').split(' ')
            social_edges.append((a[0],a[1]))

        
        for size in [128]: #50, 100, 200

            model= lineEmb( edge_file,  social_edges, name,  emb_size= size, alpha=5, epoch=10, batch_size=256, shuffel=True)
        
            embeddings= model.train(nb_labels)
        
        print('\n')
        
    node_classification(embeddings, "cora-label.txt", "cora_GEMSEC", 128)
    plot_embeddings(embeddings, "cora-label.txt", "cora_GEMSEC")
    
