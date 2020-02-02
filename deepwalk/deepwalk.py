import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from nltk.cluster import KMeansClusterer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

''' 
Run the command below to generate embeddings using Deepwalk (Library: https://github.com/phanein/deepwalk): 
deepwalk --format edgelist --input cora-edgelist.txt --max-memory-data-size 0 --number-walks 80 
--representation-size 128 --walk-length 40 --window-size 10 --workers 1 --output cora-edgelist.embeddings
'''
model = KeyedVectors.load_word2vec_format(r'cora-edgelist.embeddings', binary=False)
X = model[model.vocab]

NUM_CLUSTERS = 5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)


def vis_2D(Embedding, cluster):
    pca = PCA(n_components=2)
    pca = pca.fit_transform(Embedding)
    principalDf = pd.DataFrame(data=pca, columns=['Principal Component 1', 'Principal Component 2'])
    principalDf["y"] = cluster
    plt.figure(figsize=(8, 8))
    plt.title("Embedding's visualization (cora dataset)")
    sns.scatterplot(
        x="Principal Component 1", y="Principal Component 2",
        hue="y",
        palette=sns.color_palette("husl", 5),
        data=principalDf,
        legend="full"
    )
    plt.show()


def vis_tsne(Embedding, cluster):
    tsne = TSNE(n_components=2)
    tsne = tsne.fit_transform(Embedding)
    principalDf = pd.DataFrame(data=tsne, columns=['Principal Component 1', 'Principal Component 2'])
    principalDf["y"] = cluster
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="Principal Component 1", y="Principal Component 2",
        hue="y",
        palette=sns.color_palette("husl", 5),
        data=principalDf,
        legend="full"
    )
    plt.show()


vis_2D(X, assigned_clusters)
vis_tsne(X, assigned_clusters)
