#from eval import node_classification, plot_embeddings, link_prediction

import pickle

from sklearn.manifold import TSNE

from lineEmb import *
from Classifier import *
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
        
        # for i in range(1,5):
        #     print(i, embeddings[i])

            #node_classification( embeddings, label_file, name, size)
            #link_prediction(edge_file,  embeddings, name, size)
            #plot_embeddings( embeddings, label_file, name)
                
    # plt.plot(final_loss_list)
    # plt.ylabel('loss')
    # plt.show()

    node_classification(embeddings, "cora-label.txt", "cora_GEMSEC", 128)
    plot_embeddings(embeddings, "cora-label.txt", "cora_GEMSEC")
    


    # text_file = open("cluster.txt", "r")
    # lines = text_file.readlines()
    # cluster = [0 for i in range(2708)]
    # for line in lines:
    #     try:
    #         ll = line.strip().split(" ")
    #         cluster[int(ll[0])] = int(ll[1])
    #     except:
    #         line
    # text_file.close()
    # cluster = np.array(cluster)
    # Embedding = embeddings.values()
    # Embedding = list(Embedding)


    # def vis_2D(Embedding,cluster):
    #     pca = PCA(n_components=2)
    #     pca = pca.fit_transform(Embedding)
    #     principalDf = pd.DataFrame(data=pca, columns=['Principal Component 1', 'Principal Component 2'])
    #     principalDf["y"] = cluster
    #     plt.figure(figsize=(8, 8))
    #     plt.title("Embedding's visualization (cora dataset)")
    #     sns.scatterplot(
    #     x="Principal Component 1", y="Principal Component 2",
    #     hue="y",
    #     palette=sns.color_palette("husl", nb_labels),
    #     data=principalDf,
    #     legend="full"
    #      )
    #     plt.show()

    # def vis_tsne(Embedding, cluster):
    #     tsne = TSNE(n_components=2)
    #     tsne = tsne.fit_transform(Embedding)
    #     principalDf = pd.DataFrame(data=tsne, columns=['Principal Component 1', 'Principal Component 2'])
    #     principalDf["y"] = cluster
    #     plt.figure(figsize=(8, 8))
    #     sns.scatterplot(
    #         x="Principal Component 1", y="Principal Component 2",
    #         hue="y",
    #         palette=sns.color_palette("husl", nb_labels),
    #         data=principalDf,
    #         legend="full"
    #     )
    #     plt.show()


    # vis_2D(Embedding,cluster)
    # vis_tsne(Embedding,cluster)
