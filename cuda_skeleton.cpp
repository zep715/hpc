#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
using namespace std;


void parse_metadata(char *file, string &csv, int &N, int &M) {
    ifstream metadata(file, ios::in);
    metadata >> csv >> N >> M;
    metadata.close();
}

void parse_line(string &line, int *transaction) {
    stringstream ss(line);
    int value;
    while(ss >> value) {
        transaction[value] = 1;
        if(ss.peek() == ',')
            ss.ignore();
    }

}
void parse_transactions(string &file, int **transactions) {
    ifstream csv(file, ios::in);
    string line;
    int i = 0;
    while(getline(csv,line)) 
        parse_line(line, transactions[i++]);
    csv.close();
}

void get_1itemset(int **transactions, int N, int M, float min_sup) {
    int *counter = new int[M]();
    for (int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            counter[j] += transactions[i][j];
        }
    }
    float temp;
    for (int i = 0; i < M; i++) {
        temp = ((float)counter[i])/N;
        if (temp > min_sup)
            cout << i << endl;
    }

    delete[] counter;
}

void join_set() {

}

void itemsets_with_min_support() {

}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "/apriori input_metadata min_support, min_confidence" << endl;
        return 1;
    }
    int N, M;
    string csv;
    float min_sup = atof(argv[2]);
    float minconf = atof(argv[3]);
    
    parse_metadata(argv[1], csv, N, M);
    M++;
    int **transactions = new int*[N];
    for (int i = 0; i < N; i++) {
        transactions[i] = new int[M];
        //new int[M]() fa bloccare il pc
        memset(transactions[i], 0, M);
    }
    parse_transactions(csv, transactions);
    get_1itemset(transactions, N, M, min_sup);
    
    for (int i = 0; i < N; i++)
        delete[] transactions[i];
    delete[] transactions;
    return 0;
}