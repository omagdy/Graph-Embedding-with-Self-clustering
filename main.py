#from eval import node_classification, plot_embeddings, link_prediction

import pickle

from sklearn.manifold import TSNE

from lineEmb import *
import os
import numpy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
if __name__=='__main__':
    
    for name in ['cora-edgelist']:

        edge_file= '%s.txt'%name
        #label_file= './%s/%s-label.txt'%(name, name)
        
        
        f_social= open(edge_file, 'r')
        
        nb_labels=5
        social_edges=[]
        
        for line in f_social:
            
            a=line.strip('\n').split(' ')
            social_edges.append((a[0],a[1]))

        
        for size in [128]: #50, 100, 200

            model= lineEmb( edge_file,  social_edges, name,  emb_size= size, alpha=5, epoch=6, batch_size=128, shuffel=True)
        
            embeddings= model.train(nb_labels)
        
        print('\n')
        
        with open('embeddings.pickle', 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
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
    
    text_file = open("cluster.txt", "r")
    lines = text_file.readlines()
    cluster = [0 for i in range(2708)]
    for line in lines:
        try:
            ll = line.strip().split(" ")
            cluster[int(ll[0])] = int(ll[1])
        except:
            line
    text_file.close()
    cluster = np.array(cluster)
    Embedding = embeddings.values()
    Embedding = list(Embedding)

    def vis_3D (Embedding,cluster) :
        pca = PCA(n_components=3)
        pca = pca.fit_transform(Embedding)
        principalDf = pd.DataFrame(data=pca, columns=['one', 'two','three'])
        principalDf["y"] = cluster
        fig = plt.figure(figsize=(16, 10))
        ax = Axes3D(fig)
        ax.scatter(principalDf['one'], principalDf['two'], principalDf['three'], c=principalDf['y'], marker='o')

        plt.show()

    def vis_2D(Embedding,cluster):
        pca = PCA(n_components=2)
        pca = pca.fit_transform(Embedding)
        principalDf = pd.DataFrame(data=pca, columns=['one', 'two'])
        principalDf["y"] = cluster
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
        x="one", y="two",
        hue="y",
        palette=sns.color_palette("husl", 5),
        data=principalDf,
        legend="full"
         )
        plt.show()

    def vis_tsne(Embedding, cluster):
        tsne = TSNE(n_components=2)
        tsne = tsne.fit_transform(Embedding)
        principalDf = pd.DataFrame(data=tsne, columns=['one', 'two'])
        principalDf["y"] = cluster
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x="one", y="two",
            hue="y",
            palette=sns.color_palette("husl", 5),
            data=principalDf,
            legend="full"
        )
        plt.show()


    vis_2D(Embedding,cluster)
    vis_3D(Embedding,cluster)
    vis_tsne(Embedding,cluster)
