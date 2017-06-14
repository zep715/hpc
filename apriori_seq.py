from collections import Counter
import sys

def get_1itemset(transactions, support):
	counter = Counter()
	for trans in transactions:
		counter.update(trans)
	#print counter
	#temp = set(item for item in counter if float(counter[item])/len(transactions) >= support)
	#return set(frozenset([i]) for i in temp)
	return [(x,float(y)/len(transactions)) for x,y in counter.iteritems() if float(y)/len(transactions) >= support]

def itemsets_min_support(candidates, transactions, support):
	counter = Counter()
	for trans in transactions:
		for c in candidates:
			if c.issubset(set(trans)):
				counter[c] += 1
	return [(x,float(y)/len(transactions)) for x,y in counter.iteritems() if float(y)/len(transactions) >= support]

def load_transactions(name):
	fp = open(name, "r")
	return [line.rstrip().split(",") for line in fp]

def joinSet(s, l):
	#temp = set(map(list,itemSet))
	temp = [frozenset([x]) for x in s]
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in temp for j in temp if len(i.union(j)) == l])
	

if __name__== "__main__":
	min_sup = float(sys.argv[1])
	#min_conf = float(sys.argv[2])
	large_itemsets = list()
	transactions=load_transactions("tesco.csv")
	first_itemset = get_1itemset(transactions, min_sup)
	large_itemsets += first_itemset
	cands = [x[0] for x in first_itemset]
	k = 2
	while cands:
		cands = joinSet(cands,k)
		cands = itemsets_min_support(cands, transactions, min_sup)
		large_itemsets += cands
		cands = [x[0] for x in cands]
		k+=1
	print large_itemsets
	
	
