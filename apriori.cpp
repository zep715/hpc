#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

using namespace std;

ostream &operator<<(ostream &os, set<string> &s) {
	for (string i : s) {
		os << i << " ";
	}
    	return os;
}

ostream &operator<<(ostream &os, set<set<string> > &s) {
	for (set<string> i: s) {
		os << i << endl;
	}
	return os;
}


void load_transactions(ifstream &file, vector<set<string> > &transactions) {
	string s;
	while (file >> s) {
		set<string> t;
		string::iterator current = s.begin();
		for (string::iterator i = s.begin(); i != s.end(); i++) {
		    if (*i == ',') {
		        t.insert(string(current, i));
		        current = i+1;
		    }
		    if (i == s.end()-1) {
		        t.insert(string(current, i+1));
		    }
		}
		transactions.push_back(t);
    	}

}

void get_1itemset(vector<set<string> > &transactions,float min_sup,set<set<string> > &candidates) {

	map<string, float> counter;
	for (set<string> i : transactions) {
		for(string l : i) {
		    if(!counter.count(l)) {
			counter[l] = 1.0;
		    } else {
			counter[l] +=1.0;
		    } 
		}
	}
	for(auto& kv : counter) {
		float support = kv.second / transactions.size();
		kv.second = support;
	}
	for (auto& x : counter) {
		if (x.second >= min_sup) {
			set<string> temp;
			temp.insert(x.first);
			candidates.insert(temp);
		}
	}



}
void join_set(set<set<string> > &s, int l, set<set<string> > &cands) {
	cands.clear();
	set<string> temp;
	for (set<string> i: s) {
		for(set<string> j:s) {
			temp.clear();
			temp = i;
			temp.insert(j.begin(), j.end()); //set union
			if (temp.size() == l) {
				cands.insert(temp);
			}
			
		}
	}

}
//s.issubset(t)
//da migliorare
template<typename T>
bool is_subset(T s_begin, T s_end, T t_begin, T t_end) {
	bool result = true;
	for (T it = s_begin; it != s_end; it++) {
		bool temp = false;
		for (T it2 = t_begin; it2 != t_end; it2++) {
			if (*it == *it2) {
				temp = true;
				break;
			}
		}
		result &= temp;
	}
	
	return result;
}
void itemsets_min_support(vector<set<string> > &transactions, set<set<string> > &candidates,  map<set<string>, float> &counter, float min_sup) {
	counter.clear();
	for (set<string>  t : transactions) {
		for (set<string>  c: candidates) {
			if(is_subset(c.begin(), c.end(), t.begin(), t.end())) {
				if(!counter.count(c)) {
					counter[c] = 1.0;
				} else {
					counter[c] += 1.0;
				}
			}
		}
	}
	for (auto& kv : counter) {
		float support = kv.second / transactions.size();
		kv.second = support;

	}
	
	
	for (auto it = counter.cbegin(), next_it = counter.cbegin(); it != counter.cend(); it = next_it) {
		 next_it = it; ++next_it;
		if(it->second < min_sup) {
			counter.erase(it);
		}
	}
}


int main(int argc, char *argv[]) {
	if (argc < 4) {
		cout << "usage: ./apriori file support confidence" << endl;
		return 1;
	}
	float min_sup = atof(argv[2]);
	float min_conf = atof(argv[3]);
	vector<set<string> > transactions;
	set< set<string> > large_itemsets;
	
	ifstream input(argv[1], ios::in);
	load_transactions(input, transactions);
	
	set<set<string> >candidates;
	get_1itemset(transactions, min_sup, candidates);
	large_itemsets.insert(candidates.begin(), candidates.end());
	set<set<string> >current_litemset = candidates;
	map<set<string>, float> counter;

	for (int k = 2; !current_litemset.empty(); k++) {
		join_set(candidates,k, current_litemset);
		itemsets_min_support(transactions,current_litemset,counter,min_sup);
		current_litemset.clear();
		for (auto &kv : counter) {
			current_litemset.insert(kv.first);
			
		}
		large_itemsets.insert(current_litemset.begin(), current_litemset.end());
		candidates.clear();
		candidates = current_litemset;
		
		
	}
	cout << large_itemsets << endl;

}
