import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time

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
	#print("Building sparse matrix of size: %d" % n)

	# instantiate sparse transaction matrix
	#T = sparse.lil_matrix((n,n))

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
		snd = int(items[0])		# sender
		rcv = int(items[1])		# receiver
		n_trans = int(items[2])	# number of transactions

		# add to sparse matrix
		#T[snd, rcv] = n_trans

		# add to digraph
		G.add_nodes_from([snd, rcv])
		G.add_edge(snd,rcv, n=n_trans)

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

def compute_path_lengths(G, addr):
	idx = 0
	n = addr.shape[0]
	d = np.empty(n, dtype=int)

	# calculate each distance, use nan if unconnected
	for i,j in zip(addr[:,0], addr[:,1]):
		# output progress
		prog = 100*(idx+1)/n
		print('Computing path lengths: %.1f%% complete\r' % prog, end='')
		# get path length
		try: d[idx] = nx.shortest_path_length(G, i, j)
		except nx.NetworkXNoPath: d[idx] = -1
		# increment
		idx += 1

	print()
	return d

def stats(G, addr, mask):
	"""
	Analyze features of a set of pairs: path lengths, in/out degrees
	Plot histograms of each for the two sets determined by mask
	"""

	#########################
	# Path length histogram
	#########################

	# get distribution of distances between future interacting and non-interacting pairs
	d = compute_path_lengths(G, addr)

	fig = plt.figure(1)
	plt.hist(d[mask], 30,
		normed=True, color='r', histtype='stepfilled', alpha=0.25)
	plt.hist(d[~mask], 30,
		normed=True, color='c', histtype='stepfilled', alpha=0.25)
	plt.xlabel('Distance between addresses in training set')
	plt.ylabel('Probability density')
	beautify(fig)

	#########################
	# Degree histograms
	#########################

	bins = np.logspace(0,3,20)

	# Out-degree
	out_deg_yes = list(G.out_degree(addr[mask, 0]).values())
	out_deg_no = list(G.out_degree(addr[~mask, 0]).values())

	fig = plt.figure(2)
	plt.hist(out_deg_yes, bins,
		normed=True, color='r', histtype='stepfilled', alpha=0.25)
	plt.hist(out_deg_no, bins,
		normed=True, color='c', histtype='stepfilled', alpha=0.25)
	plt.xlabel('Out-degree of sender')
	plt.ylabel('Probability density')
	beautify(fig)
	plt.gca().set_xscale('log')

	# In-degree
	in_deg_yes = list(G.in_degree(addr[mask, 1]).values())
	in_deg_no = list(G.in_degree(addr[~mask, 1]).values())

	fig = plt.figure(3)
	plt.hist(in_deg_yes, bins,
		normed=True, color='r', histtype='stepfilled', alpha=0.25)
	plt.hist(in_deg_no, bins,
		normed=True, color='c', histtype='stepfilled', alpha=0.25)
	plt.xlabel('In-degree of receiver')
	plt.ylabel('Probability density')
	beautify(fig)
	plt.gca().set_xscale('log')

	# show
	plt.show()

def random_pairs(G, n=10000, frac=0.1):
	"""
	Select n random pairs of points, with the proportion of transactions
	given by frac.  Calculate a set of features for each pair and return
	"""
	# n x 4 feature vector to build
	# transaction / distance / out_deg / in_deg
	features = np.empty((n,5), dtype=int)
	mask = np.empty(n, dtype=bool)
	idx = 0

	# shuffle the array of nodes
	order = np.random.permutation(nx.nodes(G))
	for snd in order:
		# output progress
		prog = 100*(idx+1)/n
		#print('Considering random pairs: %.1f%% complete\r' % prog, end='')

		# decide transaction or no transaction
		if idx < n*frac:
			# choose receiving address from neighbors
			try: rcv = np.random.choice(G.successors(snd))
			except ValueError: continue # catch empty successor list
			# update mask
			mask[idx] = True
			# remove link
			G.remove_edge(snd, rcv)
		else:
			# pick a random node not in the successors of snd
			rcv = np.random.choice(nx.nodes(G))
			while rcv in G.successors(snd):
				rcv = np.random.choice(nx.nodes(G))
			# update mask
			mask[idx] = False
		t = time.clock()
		# caluclate stats without link
		try: dist = nx.shortest_path_length(G, snd, rcv)
		except nx.NetworkXNoPath: dist = -1	# catch unconnected nodes

		d_out = G.out_degree(snd)
		d_in = G.in_degree(rcv)
		features[idx, :] = [snd, rcv, dist, d_out, d_in]

		# re-add link if neceesary
		if idx < n*frac: G.add_edge(snd, rcv)
		elapsed = time.clock() - t
		print('Time of inner loop: %.4f\r' % (time.clock() - t), end='')
		# increment and break if necessary
		idx += 1
		if idx >= n: break

	return mask, features

def random_pairs2(G, n=10000, frac=0.1):
	features = np.empty((n,5), dtype=int)
	mask = np.empty(n, dtype=bool)
	idx = 0

	# PICK RANDOM TRANSACtING PAIR

	# PICK RANDOM NON-TRANSACTING PAIR

if __name__ == '__main__':
	# parse transactions and create graph
	G = parse_train()
	print('Number of nodes: %d' % nx.number_of_nodes(G))
	print('Number of edges: %d' % nx.number_of_edges(G))

	# parse future transactions
	addr, fut_mask = parse_test()

	# TODO: COME UP WITH A WAY OF FOLDING THE TRAINING DATA INTO TEST/TRAIN SETS
	# IDEAS: remove some transactions, see if they can be predicted from the rest
	# -- how to choose which transactions to predict?
	# -- dealing with overrepresented true values in test set

