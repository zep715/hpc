from mpi4py import MPI
from collections import Counter
from circular import CircularList
import sys

def joinSet(s, l):
	return set([i.union(j) for i in s for j in s if len(i.union(j)) == l])

def print_litemsets(litemsets):
	print "Large itemsets with minimum support:"
	for x,y in litemsets.iteritems():
		print "(%s) with support %.3f" % (",".join(x), y)
	print "---------"

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if rank == 0:
	min_sup = float(sys.argv[1])
	min_conf = float(sys.argv[2])
	large_itemsets = list()
	fp = open("tesco.csv")
	clist = CircularList(size-1)
	n_transactions = 0
	for line in fp:
		clist.append(line.rstrip().split(","))
		n_transactions +=1
	for i in range(1,size):
		comm.send(clist[i-1], dest=i)
	global_counter = Counter()
	for i in range(1,size):
		partial_counter = comm.recv()
		global_counter += partial_counter
	first_itemset =[(x,float(y)/n_transactions) for x,y in global_counter.iteritems() if float(y)/n_transactions >= min_sup]
	large_itemsets += [(frozenset([x[0]]),x[1]) for x in first_itemset]
	cands = set(frozenset([x[0]]) for x in first_itemset)
	k = 2
	while True:
		cands = joinSet(cands,k)
		for i in range(1,size):
			comm.send(cands, dest=i)
		if cands == set():
			break
		current_counter = Counter()
		for i in range(1,size):
			partial_counter = comm.recv()
			current_counter += partial_counter
		cands=[(x,float(y)/n_transactions) for x,y in current_counter.iteritems() if float(y)/n_transactions >= min_sup]
		large_itemsets += cands
		cands = [x[0] for x in cands]
		
		k+=1
	large_itemsets = dict(large_itemsets)
	print_litemsets(large_itemsets)
else:
	partial_transactions = comm.recv(source=0)
	counter = Counter()
	for x in partial_transactions:
		counter.update(x)
	comm.send(counter, dest=0)
	while True:
		cands = comm.recv(source=0)
		if cands == set():
			break
		counter = Counter()
		for t in partial_transactions:
			for c in cands:
				if c.issubset(set(t)):
					counter[c] += 1
		comm.send(counter, dest=0)
