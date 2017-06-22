#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstdlib>
#include <algorithm>
using namespace std;


void print_set(vector<vector<int> > &s, const string title) {
	cout << title << endl;
	for(vector<int> &x : s) {
		for (int y : x) { 
			cout << y << " ";
		}
		cout << ";";
	}
	cout << endl;

}

void print_largeitemset(map<vector<int>, float> &large_itemsets) {
	cout << "Large itemsets with min support:" << endl;
	for (auto &kv : large_itemsets) {
		cout << "(";
		vector<int>::const_iterator it;
		for (it = kv.first.begin(); it != kv.first.end()-1; ++it)  {
			cout << *it << ",";
		}
	
		cout << *(it) << "): " << kv.second << endl;
	}
	cout << "------------------------" << endl;

}
void load_transactions(ifstream &input, vector<vector<int> > &transactions, int &m) {
	string s;
	vector<int> temp;
	int max = 0;
	while(input >> s) {
		temp.clear();
		string::iterator current = s.begin();
		for (string::iterator i = s.begin(); i != s.end(); i++) {
		    if (*i == ',') {
			temp.push_back(atoi(string(current,i).c_str()));
		        current = i+1;
		    }
		    if (i == s.end()-1) {
			temp.push_back(atoi(string(current,i+1).c_str()));
		    }
		}
		transactions.push_back(temp);
		if (temp.back() >= max)
			max = temp.back();
	}
	m = max;


}



void get_1itemset(int max_id, float min_sup, vector<vector<int> > &transactions, vector<vector<int> > &first_itemset, map<vector<int>, float> &large_itemsets) {
	int *counter = new int[max_id]();
	unsigned int N = transactions.size();
	for(unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < transactions[i].size(); j++) {
			counter[transactions[i][j]] += 1;
		}
	}
	vector<int> temp;
	for (int i = 0; i < max_id; i++) {
		temp.clear();
		if ( (((float)counter[i]) / N) >= min_sup) {
			temp.push_back(i);
			first_itemset.push_back(temp);
			large_itemsets[temp] = ((float)counter[i]) / N;
		}
			
	}
	delete[] counter;
}

void itemsets_min_support(vector<vector<int> > &transactions, vector<vector<int> > &candidates, float min_sup, vector<vector<int> > &result, map<vector<int>, float> &large_itemsets) {
	int *counter = new int[candidates.size()]();
	unsigned int N = transactions.size();
	for (unsigned int i = 0; i < transactions.size(); i++) {
		for (unsigned int j = 0; j < candidates.size(); j++) {
			//if every element in cadidates[j] is in transactions[i], then counter[j]++
			bool not_found = false;			
			for(unsigned int k = 0; k < candidates[j].size(); k++) {
				if (!binary_search(transactions[i].begin(), transactions[i].end(), candidates[j][k])) {
					not_found = true;
					break;
				}
			}
			if (!not_found) {
				counter[j]++;
			}	
		}
	}
	for (unsigned int i = 0; i < candidates.size(); i++) {
		if ( (((float)counter[i]) / N) >= min_sup) {
			result.push_back(candidates[i]);
			large_itemsets[candidates[i]] = ((float)counter[i]) / N;
		}
	}
	delete[] counter;
}


void join_set( vector<vector<int> > &itemset, int k, vector<vector<int> > &candidates) {
	if (k == 2) {
		vector<int> temp;
		for (unsigned int i = 0; i < itemset.size(); i++) {
			for (unsigned int j = i+1; j < itemset.size(); j++) {
				temp.clear();
				temp.push_back(itemset[i][0]);
				temp.push_back(itemset[j][0]);
				candidates.push_back(temp);
			}
		}
		return;
	}
	int len = k-2;
	vector<int> temp;
	for (unsigned int i = 0; i < itemset.size(); i++) {
		for (unsigned int j = 0; j < itemset.size(); j++) {
			bool joinable = true;
			temp.clear();
			for (int k = 0; k < len; k++) {
				if (itemset[i][k] != itemset[j][k]) {
					joinable = false;
					break;
				}
			}
			if (!joinable)
				continue;
			if (itemset[i].back() == itemset[j].back()) {
				continue;
			}
			temp.assign( itemset[i].begin(), itemset[i].end()-1 );
			if (itemset[i].back() >= itemset[j].back()) {
				temp.push_back(itemset[j].back());
				temp.push_back(itemset[i].back());
			} else {
				temp.push_back(itemset[i].back());
				temp.push_back(itemset[j].back());
			}
			candidates.push_back(temp);
			
		}
	}

}
int main(int argc, char *argv[]) {
	if (argc != 4) {
		cout << "usage: ./apriori file support confidence" << endl;
		return 1;
	}
	int max_id;
	float min_sup, min_conf;
	vector<vector<int> > transactions, first_itemset, candidates, current_itemset;
	map<vector<int>, float> large_itemsets;
	ifstream input;
	
	min_sup = atof(argv[2]);
	min_conf = atof(argv[3]);
	input.open(argv[1], ios::in);
	
	load_transactions(input, transactions, max_id);
	input.close();
	
	get_1itemset(max_id, min_sup, transactions, first_itemset, large_itemsets);
	current_itemset = first_itemset;
	//print_set(first_itemset, "fisrst itemset");
	for (int k = 2; ; k++) {
		candidates.clear();
		join_set(current_itemset, k, candidates);
		//print_set(candidates,"candidates");
		if (candidates.empty())
			break;
		current_itemset.clear();
		itemsets_min_support(transactions, candidates, min_sup, current_itemset, large_itemsets);
		//print_set(current_itemset,"candidates with min support");
		
	}
	candidates.clear();
	print_largeitemset(large_itemsets);
	for (const auto &kv: large_itemsets) {
		candidates.push_back(kv.first);
	}
	for (unsigned int i = 0; i < candidates.size(); i++) {
		
	}
}
