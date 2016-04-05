import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import sklearn.decomposition.LatentDirichletAllocation
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_curve, auc
import os
import sys

os.chdir('data')


dftrain = pd.read_csv('txTripletsCounts.txt',
                      header=None,
                      index_col=None,
                      sep=' ',
                      names=['sender','receiver','transaction'])
dftrain['transaction'].describe()

n_sender = dftrain['sender'].unique().size
n_receiver = dftrain['receiver'].unique().size
n_nonmissing = dftrain['transaction'].count()
sparsity = 1 - n_nonmissing*1.0/n_sender/n_receiver

print "n_sender=%i" % n_sender
print "n_receiver=%i" % n_receiver
print "n_nonmissing=%i"% n_nonmissing
print "sparsity=%2.6f"% sparsity




dftest = pd.read_csv('testTriplets.txt',
                     header=None,
                     index_col=None,
                     sep=' ',
                     names=['sender','receiver','transaction'])
dim = max((dftrain['sender'].max(),
           dftrain['receiver'].max(),
           dftest['sender'].max(),
           dftest['receiver'].max()))
dim += 1
train_csr = csr_matrix((dftrain['transaction'],(dftrain['sender'],dftrain['receiver'])),
                       shape=(dim,dim),
                       dtype=float)




#Matrix factorization techniques:

label = dftest['transaction']

#Scipy SVD
u, s, vt = svds(train_csr, k=10, tol=1e-10, which = 'LM')
pred1 = [np.sum(u[row['sender'],:] * s * vt[:,row['receiver']])
        for index,row in dftest.iterrows()]

#plotroc(label, pred1, "svd")


#Scikit Learn NMF, Takes ~2 min
nmf_model = NMF(n_components = 20, tol=1e-10, max_iter=2, nls_max_iter=30)
W = nmf_model.fit_transform(train_csr)
H = nmf_model.components_
pred2 = [np.sum(W[row['sender'],:] * H[:,row['receiver']])
        for index,row in dftest.iterrows()]

#plotroc(label, pred2, "nmf")


#Random test
ra = np.random.rand(u.shape[0], u.shape[1])
rb = np.random.rand(vt.shape[0], vt.shape[1])
predr = [np.sum(ra[row['sender'],:] * rb[:,row['receiver']])
        for index,row in dftest.iterrows()]

#plotroc(label, predr, "random")



os.chdir("../")

parr = [pred1, pred2, predr]
narr = ["SVN", "NMF", "Random"]

for x in range(3):
  pred = parr[x]
  fpr, tpr, thresholds = roc_curve(label, pred)
  roc_auc = auc(fpr, tpr)
  print "Area under the ROC curve : %f" % roc_auc
  matplotlib.rcParams['figure.figsize'] = (10, 10)
  #plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot(fpr, tpr, label=narr[x] % roc_auc)


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right", fontsize=16)
#plt.show()
plt.savefig("ROC.png")





















#Scikit Learn SVD, Takes ~1 min
#U, S, VT = randomized_svd(train_csr, n_components = 20, n_iter=10)
#W = svd_model.fit_transform(train_csr)
#H = svd_model.components_
#pred3 = [np.sum(U[row['sender'],:] * S * VT[:,row['receiver']])
#        for index,row in dftest.iterrows()]
#pred3 = [np.sum(W[row['sender'],:] * H[:,row['receiver']])
#        for index,row in dftest.iterrows()]
#
#plotroc(label, pred3)


















import matplotlib.pyplot as plt
import matplotlib

def plotroc(label, pred, fn):
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    matplotlib.rcParams['figure.figsize'] = (10, 10)
    plt.plot(fpr, tpr, color='magenta', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(fn + ".png")







