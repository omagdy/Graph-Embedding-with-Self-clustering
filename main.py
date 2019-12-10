# from eval import node_classification, plot_embeddings, link_prediction
from lineEmb import *
if __name__ == '__main__':

    for name in ['airport']:

        edge_file = '%s.txt' % name
        # label_file= './%s/%s-label.txt'%(name, name)

        f_social = open(edge_file, 'r')

        social_edges = []

        for line in f_social:
            a = line.strip('\n').split(' ')
            social_edges.append((a[0], a[1]))

        for size in [8]:  # 50, 100, 200

            model = lineEmb(edge_file, social_edges, name, emb_size=size, alpha=5, epoch=6, batch_size=128,
                            shuffel=True)

            embeddings = model.train()

        print('\n')

        for i in range(1, 5):
            print(i, embeddings[i])

            # node_classification( embeddings, label_file, name, size)
            # link_prediction(edge_file,  embeddings, name, size)
            # plot_embeddings( embeddings, label_file, name)

