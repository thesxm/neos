#include <iostream>
#include "../include/evolution.hpp"
#include "../include/neural.hpp"

using namespace std;
using NEURAL::Network;
using EVOLUTION::generate_initial_population;

void printVector(vector<float> v) {
    cout << "(";
    for (auto c: v) cout << c << " ";
    cout << ")" << endl;

    return;
}

int main(int argc, char** args) {
    srand(time(0));

    int n = 10;
    vector<Network> population = generate_initial_population(n, 3, 2, {3, 3, 2});
    vector<float> inp = {1, 1, 10};

    for (int i = 0; i < n; i++)
        printVector(population[i] << inp);
    
    vector<Network> next_generation;
    for (int i = 0; i < n; i += 2)
        next_generation.push_back(population[i] * population[i + 1]);
    
    cout << "\nNext Generation\n";
    for (int i = 0; i < next_generation.size(); i++)
        printVector(next_generation[i] << inp);

    return 0;
}