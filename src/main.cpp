#include <iostream>
#include <cmath>
#include "../include/evolution.hpp"
#include "../include/neural.hpp"

using namespace std;
using EVOLUTION::Environment;
using NEURAL::Network;

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

    Environment env = Environment(
        500,
        .5,
        .25,
        2,
        1,
        new vector<int>({1}),
        &fitness_function);

    vector<float> *inp_1 = new vector<float>({1, 1});
    vector<float> *out_1 = new vector<float>({2});
    vector<float> *inp_2 = new vector<float>({5, 9});
    vector<float> *out_2 = new vector<float>({14});
    vector<float> *inp_3 = new vector<float>({-9, 3});
    vector<float> *out_3 = new vector<float>({-6});

    vector<vector<float> *> *inps = new vector<vector<float> *>({inp_1, inp_2, inp_3});
    vector<vector<float> *> *outs = new vector<vector<float> *>({out_1, out_2, out_3});

    auto res = env.evolve_n_generations(10000, inps, outs);

    // for (int i = 0; i < res->size(); i++) {
    //     cout << "MODEL " << i << " (" << *(res->at(i)) << ") LAYER COUNT " << res->at(i)->layer_count() << endl;
    //     for (int j = 0; j < inps->size(); j++) {
    //         vector<float>* network_out = *(res->at(i)) << *(inps->at(j));

    //         cout << "\tINPUT " << j << " ";
    //         printVector(*(inps->at(j)));
    //         cout << " ";
    //         printVector(*network_out);
    //         cout << " " << fitness_function(network_out, outs->at(j));
    //         cout << endl;
    //     }
    // }

    return 0;
}