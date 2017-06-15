from collections import Counter
from itertools import chain,combinations
import sys

def get_1itemset(transactions, support):
	counter = Counter()
	for trans in transactions:
		counter.update(trans)
	return [(x,float(y)/len(transactions)) for x,y in counter.iteritems() if float(y)/len(transactions) >= support]

def itemsets_min_support(candidates, transactions, support):
	counter = Counter()
	for trans in transactions:
		for c in candidates:
			if c.issubset(set(trans)):
				counter[c] += 1
	return [(x,float(y)/len(transactions)) for x,y in counter.iteritems() if float(y)/len(transactions) >= support]

def getSupport(item, transactions):
	count = 0
	for t in transactions:
		if item.issubset(set(t)):
			count +=1
	return float(count)/len(transactions)

def load_transactions(name):
	fp = open(name, "r")
	return [line.rstrip().split(",") for line in fp]

def joinSet(s, l):
	return set([i.union(j) for i in s for j in s if len(i.union(j)) == l])

def subsets(arr):
    	temp = chain(*[combinations(arr, i + 1) for i in range(len(arr))])
	return map(frozenset, [x for x in temp])


def print_rules(rules):
	print "Rules with minimum confidence:"
	for x in rules:
		print "(%s) => (%s) with confidence %.3f" % (",".join(x[0]),",".join(x[1]),x[2])
	print "---------"
def print_litemsets(litemsets):
	print "Large itemsets with minimum support:"
	for x,y in litemsets.iteritems():
		print "(%s) with support %.3f" % (",".join(x), y)
	print "---------"
if __name__== "__main__":
	min_sup = float(sys.argv[1])
	min_conf = float(sys.argv[2])
	large_itemsets = list()
	transactions=load_transactions("tesco.csv")
	first_itemset = get_1itemset(transactions, min_sup)
	large_itemsets += [(frozenset([x[0]]),x[1]) for x in first_itemset]
	cands = set(frozenset([x[0]]) for x in first_itemset)
	k = 2
	while cands:
		cands = joinSet(cands,k)
		cands = itemsets_min_support(cands, transactions, min_sup)
		large_itemsets += cands
		cands = [x[0] for x in cands]
		k+=1
	large_itemsets = dict(large_itemsets)
	print_litemsets(large_itemsets)
	rules = list()
	for l in large_itemsets.keys():
		for a in subsets(l):
			remain = l.difference(a)
			if len(remain) > 0:
				confidence = large_itemsets[l]/getSupport(a, transactions)
				if confidence >= min_conf:
					rules.append( [a, remain, confidence])
	
	print_rules(rules)
	
