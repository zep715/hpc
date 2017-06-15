#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <set>
#include <map>

using namespace std;



int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "usage: ./apriori file support confidence" << endl;
        return 1;
    }
    float min_sup = atof(argv[2]);
    float min_conf = atof(argv[3]);
    vector<set<string> > transactions;
    ifstream input(argv[1], ios::in);
    string s;
    while (input >> s) {
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
    /*
    cout << "----" << endl;
    for (set<string> i : transactions) {
        for(string l : i) {
            cout << l << " ";
        }
        cout << endl;
    }
    */
    set<string> first_itemset;
    for(auto& kv : counter) {
        float support = kv.second / transactions.size();
        if (support < min_sup) {
            counter.erase(kv.first);
        } else {
            kv.second = support;
            first_itemset.insert(kv.first);
        }
    }
    for(auto& kv : first_itemset) {
        cout << kv << endl;
    }

}
