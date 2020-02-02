import numpy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


class TopKRanker(OneVsRestClassifier):

    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf, name):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.name = name

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)

        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)

        results['acc'] = accuracy_score(Y, Y_)
        print('-------------------')
        print(self.name, 'node calssification: ', results)
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

def read_node_label(embeddings, label_file, skip_head=False):
        fin = open(label_file, 'r')
        X = []
        Y = []
        label = {}

        for line in fin:
            a = line.strip('\n').split(' ')
            label[a[0]] = a[1]

        fin.close()
        for i in embeddings:
            X.append(i)
            Y.append(label[str(i)])

        return X, Y


def node_classification(embeddings, label_path,name, size):
        X, Y = read_node_label(embeddings, label_path )

        f_c = open('%s_classification_%d.txt' % (name, size), 'w')

        all_ratio = []

        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(" Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
            clf = Classifier(embeddings=embeddings, clf=LogisticRegression(), name=name)
            results = clf.split_train_evaluate(X, Y, tr_frac)
            avg = 'macro'
            f_c.write(name + ' train percentage: ' + str(tr_frac) + ' F1-' + avg + ' ' + str('%0.5f' % results[avg]))
            all_ratio.append(results[avg])
            f_c.write('\n')


def plot_embeddings(embeddings,label_file,name):
    X, Y = read_node_label(embeddings, label_file)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = numpy.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.legend()

    plt.savefig('%s.png' % name)  # or '%s.pdf'%name
    plt.show()