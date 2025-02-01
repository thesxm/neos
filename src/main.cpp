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
    return exp(-1 - abs(network_output->at(0) - ideal_output->at(0)));
}

int main(int argc, char **args)
{
    srand(time(0));

    auto train = read_from_file("data/train.data");
    auto test = read_from_file("data/test.data");

    auto train_inps = train->at(0);
    auto train_outs = train->at(1);
    
    auto test_inps = test->at(0);
    auto test_outs = test->at(1);

    Environment env = Environment(
        50,
        .5,
        .25,
        train_inps->at(0)->size(),
        train_outs->at(0)->size(),
        new vector<int>({10, 10, 10}),
        &fitness_function);


    auto res = env.evolve_n_generations(100, train_inps, train_outs);

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

    cout << "Maximum Mean Squared Fitness over test data: " << max_msf << endl;
    // cout << "Test outputs of best network (" << best_network << ")" << endl;
    // for (int i = 0; i < test_inps->size(); i++)
    // {
    //     cout << "\t";
    //     printVector(*test_inps->at(i));
    //     cout << " => ";
    //     printVector(*(*best_network << *test_inps->at(i)));
    //     cout << " | EXPECTED ";
    //     printVector(*test_outs->at(i));
    //     cout << endl;
    // }

    return 0;
}