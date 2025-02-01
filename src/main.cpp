#include <iostream>
#include <cmath>
#include "../include/evolution.hpp"
#include "../include/neural.hpp"
#include "../include/reader.hpp"

using namespace std;
using EVOLUTION::Environment;
using NEURAL::Network;
using READER::read_from_file;

void printVector(vector<float> v)
{
    cout << "(";
    for (auto c : v)
        cout << c << " ";
    cout << ")";

    return;
}

float fitness_function(vector<float> *network_output, vector<float> *ideal_output)
{
    float res = 0;
    for (int i = 0; i < network_output->size(); i++)
        res += pow(network_output->at(i) - ideal_output->at(i), 2);
    res = sqrt(res);

    return exp(-res) * 100 / exp(1);
}

int main(int argc, char **args)
{
    srand(time(0));

    auto train = read_from_file("src/train.data");
    auto test = read_from_file("src/test.data");

    auto train_inps = train->at(0);
    auto train_outs = train->at(1);
    
    auto test_inps = test->at(0);
    auto test_outs = test->at(1);

    Environment env = Environment(
        5000,
        .5,
        .25,
        2,
        1,
        new vector<int>({3, 3, 3}),
        &fitness_function);


    auto res = env.evolve_n_generations(500, train_inps, train_outs);

    float max_msf = 0;
    Network *best_network;
    for (auto network : *res)
    {
        float msf = 0;
        for (int i = 0; i < test_inps->size(); i++)
            msf += pow(fitness_function(*network << *test_inps->at(i), test_outs->at(i)), 2);
        msf = sqrt(msf);

        if (msf > max_msf)
        {
            max_msf = msf;
            best_network = network;
        }
    }

    cout << "Maximum Mean Squared Fitness: " << max_msf << endl;
    cout << "Outputs of best network" << endl;
    for (int i = 0; i < test_inps->size(); i++)
    {
        cout << "\t";
        printVector(*test_inps->at(i));
        cout << " => ";
        printVector(*(*best_network << *test_inps->at(i)));
        cout << endl;
    }

    while (true) {
        float a, b;

        cout << "Enter a: ";
        cin >> a;
        cout << "Enter b: ";
        cin >> b;

        float output = (*best_network << vector<float>({a, b}))->at(0);
        cout << ">> " << output << endl;
    }

    return 0;
}