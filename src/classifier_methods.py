import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import sys, os.path
# sklearn
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import sklearn.metrics as mt
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

n_features = 5

def print_stars(n=30):
    stars = ""
    for i in range(n): stars += '*'
    print(stars)

def beautify(fig):
    # Set background color to white
    fig.patch.set_facecolor((1,1,1))

    # Remove frame and unnecessary tick-marks
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(axis='y', which='both', left='off', right='off')
    plt.tick_params(axis='x', which='both', top='off', bottom='off')

def parse_train(fname='../data/txTripletsCounts.txt'):
    # open and read
    with open(fname, 'r') as file:
        lines = file.readlines()

    # how many entries are there
    n = len(lines)

    # instantiate DiGraph
    G = nx.DiGraph()

    # index for progress bar
    idx = 0
    # parse each line
    for line in lines:
        # print progress
        prog = 100*(idx+1)/n
        print("Parsing transactions: %.1f%% complete\r" % prog, end='')
        idx += 1

        # parse each item in the line
        items = line.split()
        snd = int(items[0])     # sender
        rcv = int(items[1])     # receiver
        n_trans = int(items[2]) # number of transactions

        # add to sparse matrix
        #T[snd, rcv] = n_trans

        # add to digraph
        G.add_nodes_from([snd, rcv])
        G.add_edge(snd, rcv, trans=n_trans)

    print()
    return G

def parse_test(fname='../data/testTriplets.txt'):
    # open and read
    with open(fname, 'r') as file:
        lines = file.readlines()

    n = len(lines)
    fut_mask = np.empty(n, dtype=bool)
    addr = np.empty((n,2), dtype=int)

    idx = 0
    for line in lines:
        prog = 100*(idx+1)/n
        print('Parsing test data: %.1f%% complete\r' % prog, end='')
        # split items
        items = line.split()
        snd = int(items[0])
        rcv = int(items[1])
        fut = int(items[2])

        # save
        addr[idx, :] = [snd, rcv]
        if fut == 1: fut_mask[idx] = True
        else: fut_mask[idx] = False

        idx += 1

    print()
    return addr, fut_mask

def feat_hists(feats, mask, feat_names=None, n_bins=18):
    """ Plot overlapping histograms of each feature, separating
    into two categories according to mask """
    n_feats = feats.shape[1] # number of features
    if feat_names == None:
        feat_names = ["Feature " + str(i+1) for i in range(n_feats)]

    for idx in range(n_feats):
        # instantiate plot
        fig = plt.figure(idx)

        # get range of data and generate histogram bins
        if feat_names[idx] == 'Directed distance along graph':
            bins = range(1,10)
        else:
            bins = np.logspace(0, 3, n_bins)

        # plot histograms
        plt.hist(feats[mask, idx], bins, label='Positive Class',
            normed=True, color='r', histtype='stepfilled', alpha=0.25)
        plt.hist(feats[~mask, idx], bins, label='Negative Class',
            normed=True, color='c', histtype='stepfilled', alpha=0.25)

        # figure labels
        plt.xlabel(feat_names[idx])
        plt.ylabel('Probability Density')
        plt.legend()
        beautify(fig)

        if feat_names[idx] != 'Directed distance along graph':
            plt.gca().set_xscale('log')

    # show all
    plt.show()

def random_pairs(G, n=100000, frac=0.1, saving=True):
    """ Select a training set of n send/receive pairs, with frac
    of them representing actual transactions.  Calculate stats for
    each pair using the get_features() helper.  For those send/receive
    pairs that represent actual transactions in the training data, remove
    the link from the graph before calculating the features to simulate
    transaction prediction. """
    # initialize numpy arrays for holding features
    features = np.empty((n, n_features), dtype=int)
    mask = np.empty(n, dtype=bool)

    # Grab edges and nodes
    edges_list = nx.edges(G)
    np.random.shuffle(edges_list)
    nodes_list = nx.nodes(G)

    # number of real transactions to consider
    n_true = int(n*frac)

    # pick random transacting pairs and calculate features w/o transaction
    for i in range(n_true):
        # output progress
        prog = 100*(i+1)/n
        print('Considering random pairs: %.1f%% complete\r' % prog, end='')

        # set mask to true
        mask[i] = True
        # Pick pair from list of transactions
        (snd, rcv) = edges_list[i]
        # Save edge weight and remove from graph
        n_trans = G[snd][rcv]['trans']
        G.remove_edge(snd, rcv)
        # get features
        features[i, :] = get_features(G, snd, rcv)
        # re-add link
        G.add_edge(snd, rcv, trans=n_trans)

    # Calculate stats for unconnected nodes
    for i in range(n_true, n):
        # output progress
        prog = 100*(i+1)/n
        print('Considering random pairs: %.1f%% complete\r' % prog, end='')

        # set mask to false
        mask[i] = False
        # Pick two addresses and ensure they are unconnected
        snd, rcv = np.random.choice(nodes_list, 2)
        while G.has_edge(snd, rcv):
            snd, rcv = np.random.choice(nodes_list, 2)
        # get features
        features[i, :] = get_features(G, snd, rcv)

    # save to csv file
    if saving:
        feats_and_mask = np.hstack(features, mask.astype(int))
        np.savetxt('random_pairs.txt', feats_and_mask, delimiter=',')

    print() # end line

    return features, mask

def get_features(G, snd, rcv):
    # Distance between nodes
    try: dist = nx.shortest_path_length(G, snd, rcv)
    except nx.NetworkXNoPath: dist = -1 # catch unconnected nodes

    # in- and out-degrees
    d_out = G.out_degree(snd)
    d_in = G.in_degree(rcv)

    # total number of coins sent by sender
    t_out = 0
    for scr in G.successors(snd):
        t_out += G[snd][scr]['trans']

    # total number of coins received by receiver
    t_in = 0
    for pdr in G.predecessors(rcv):
        t_in += G[pdr][rcv]['trans']

    # node connectivity
    #conn = nx.node_connectivity(G, snd, rcv)

    return [dist, d_out, d_in, t_out, t_in] #, conn]

def compile_features(G, addr, saving=True):
    n = addr.shape[0]
    feats = np.empty((n, n_features), dtype=int)

    for i in range(n):
        # print progress
        prog = 100*(i+1)/n
        print('Gathering features: %.1f%% complete\r' % prog, end='')
        # get features
        feats[i,:] = get_features(G, *addr[i,:])

    if saving:
        np.savetxt('test_feats.txt', feats, delimiter=',')

    print()
    return feats


def evaluate(model, feats, mask):
    probs = model.predict_proba(feats)
    # calcuate roc auc:
    roc = mt.roc_auc_score(mask, probs[:,1])
    # calculate precision-recall auc
    prc = mt.average_precision_score(mask, probs[:,1])

    return roc, prc, probs

def plot_curves(probs_list, names, feats, mask):
    # plot each curve
    prc = plt.figure('PRC')
    roc = plt.figure('ROC')
    prc_handles = []
    roc_handles = []

    for probs in probs_list:
        # precision-recall
        plt.figure('PRC')
        precision, recall, _ = mt.precision_recall_curve(mask, probs[:,1])
        h, = plt.plot(precision, recall)
        prc_handles.append(h)

        # receiver operating characteristic
        plt.figure('ROC')
        fpr, tpr, _ = mt.roc_curve(mask, probs[:,1])
        h, = plt.plot(fpr, tpr)
        roc_handles.append(h)

    # clean up / decorate
    beautify(prc)
    plt.figure('PRC')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curves')
    plt.legend(prc_handles, names, loc=1)

    beautify(roc)
    plt.figure('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(roc_handles, names, loc=4)

    plt.show()

def train_test_print(model, train_feats, test_feats, train_mask, test_mask):
    # fit
    model.fit(train_feats, train_mask)

    # evaluate performance
    roc, prc, probs = evaluate(model, test_feats, test_mask)
    roc_train, prc_train, _ = evaluate(model, train_feats, train_mask)

    print('ROC: %.3f  (%.3f)' % (roc, roc_train))
    print('PRC: %.3f  (%.3f)' % (prc, prc_train))

    return roc, prc, probs

def pairwise_products(feats):
    products = lambda x: np.asarray(np.mat(x).transpose() * np.mat(x)).flatten()
    return np.apply_along_axis(products, 1, feats)

if __name__ == '__main__':
    # parse future transactions (testing data)
    test_addr, test_mask = parse_test()
    n_test = test_addr.shape[0]

    # extract features from training data
    if os.path.isfile('random_pairs.txt'):
        data = np.genfromtxt('random_pairs.txt', delimiter=',')
        train_feats = data[:,:-1]
        train_mask = data[:,-1].astype(bool)
        del(data)
    else:
        # parse transactions and create graph
        G = parse_train()
        train_feats, train_mask = random_pairs(G)

    # get features from each transaction pair
    if os.path.isfile('test_feats.txt'):
        test_feats = np.genfromtxt('test_feats.txt', delimiter=',')

    else:
        if 'G' not in locals: G = parse_train()
        test_feats = compile_features(G, test_addr)


    #############################################
    # Visualize extracted features
    #############################################

    # histograms of training data
    feat_names = ['Directed distance along graph', 'Out-degree of sender',
        'In-degree of reciever', 'Total sending transactions of sender',
        'Total receiving transactions of receiver']

    feat_hists(train_feats, train_mask, feat_names)

    # plot the subgraph represented by the training data

    # TODO

    #############################################
    # Train models on features / known assignments
    #############################################

    # replace missing (-1) values for distance with max int size
    train_feats[train_feats == -1] = sys.maxsize

    print_stars()
    print('Multinomial Naive Bayes')
    mnb = MultinomialNB(class_prior=[0.9, 0.1])
    _, _, mnb_probs = train_test_print(
        mnb, train_feats, test_feats, train_mask, test_mask)


    print_stars()
    print('Gaussian Naive Bayes')
    gnb = GaussianNB()
    gnb.class_prior_ = [0.9, 0.1]
    _, _, gnb_probs = train_test_print(
        gnb, train_feats, test_feats, train_mask, test_mask)

    print_stars()
    print('Naive Bayes with Square Kernel')
    gnb2 = GaussianNB()
    gnb2.class_prior_ = [0.9, 0.1]
    test_prods = pairwise_products(test_feats)
    train_prods = pairwise_products(train_feats)
    train_test_print(gnb2, train_prods, test_prods, train_mask, test_mask)

    print_stars()
    print('K-Nearest Neighbors, k = 25')
    knn = KNeighborsClassifier(n_neighbors=25, weights='distance')
    _, _, knn_probs = train_test_print(
        knn, train_feats, test_feats, train_mask, test_mask)

    print_stars()
    print('Random Forest')
    fst = RandomForestClassifier()
    _, _, fst_probs = train_test_print(
        fst, train_feats, test_feats, train_mask, test_mask)


    # plot roc and precision-recal curves
    all_probs = [mnb_probs, gnb_probs, knn_probs, fst_probs]
    names = ['Multinomial Naive Bayes', 'Gaussian Naive Bayes',
        'K-Nearest Neighbors', 'Random Forest']
    plot_curves(all_probs, names, test_feats, test_mask)

    # Fold the training data to optimize models?


