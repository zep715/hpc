from collections import Counter
from itertools import chain, combinations
def apriorigen(l,k):
	c_temp = set()
	for a in l:
		for b in l:
			union = a | b
			if len(union) == k:
				c_temp.add(union)
	return c_temp

def get_1itemset(transactions, support):
	counter = Counter()
	for trans in transactions:
		counter.update(trans)
	temp = set(item for item in counter if float(counter[item])/len(transactions) >= support)
	return set(frozenset([i]) for i in temp)


def load_transactions(name):
	fp = open(name, "r")
	return [line.rstrip().split(",") for line in fp]

def itemsets_support(transactions, itemsets):
    support_set = Counter()

    for trans in transactions:
        subsets = [itemset for itemset in itemsets if itemset < set(trans)]
        support_set.update(subsets)

	return support_set
def min_support_set(counter, support):

    	items = [item for item in counter if counter[item] >= support]
	return set(items)

def subsets(arr):
    	temp = chain(*[combinations(arr, i + 1) for i in range(len(arr))])
	return map(frozenset, [x for x in temp])

def getSupport(item, transactions):
	count = 0
	for t in transactions:
		if item.issubset(t):
			count +=1
	return float(count)/len(transactions)
	
def apriori(name, minsup, minconf):
	t_list = load_transactions(name)
	itemset = get_1itemset(t_list, minsup)
	k = 2
	large_itemsets = list()
	candidates = itemset
	while candidates:
		candidates = apriorigen(candidates, k)
		supported = itemsets_support(t_list, candidates)
		candidates = min_support_set(supported,minsup)
		large_itemsets += candidates
		k+=1
	rules = list()
	for l in large_itemsets:
		for a in subsets(l):
			remain = l.difference(a)
			if len(remain) > 0:
				confidence = getSupport(l, t_list)/getSupport(a, t_list)
				if confidence > minconf:
					rules.append( [a, remain, confidence])
					
	return large_itemsets,rules

def print_rules(rules):
	for r in rules:
		#print "%s => %s with confidence: %f" % (str(list(r[0])),str(list(r[1]).join(",")), r[2])
		print "%s => %s with confidence: %.2f" % (",".join(r[0]), ",".join(r[1]), r[2])
if __name__=="__main__":
	itemsets,rules = apriori("tesco.csv", 0.4, 0.4)
	print_rules(rules)
