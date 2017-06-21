#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <cstdlib>
#include <chrono>
#include <algorithm>
using namespace std;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock ;



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



void get_1itemset(int max_id, float min_sup, vector<vector<int> > &transactions, vector<vector<int> > &first_itemset) {
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
		}
			
	}
	delete[] counter;
}

void itemsets_min_support(vector<vector<int> > &transactions, vector<vector<int> > &candidates, float min_sup, vector<vector<int> > &result) {
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
		}
	}
	delete[] counter;
}

void first_candidates(vector<int> &first_itemset, vector<vector<int> > &candidates) {
	
	vector<int> temp;
	for (unsigned int i = 0; i < first_itemset.size(); i++) {
		
		for (unsigned int j = i+1; j < first_itemset.size(); j++) {
			temp.clear();
			temp.push_back(first_itemset[i]);
			temp.push_back(first_itemset[j]);			
			candidates.push_back(temp);
			
		}
	}	


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
	vector<vector<int> > transactions;
	vector<int> first_itemset;
	ifstream input;
	
	min_sup = atof(argv[2]);
	min_conf = atof(argv[3]);
	input.open(argv[1], ios::in);
	
	load_transactions(input, transactions, max_id);
	input.close();
	
	get_1itemset(max_id, min_sup, transactions, first_itemset);
	
	
}
