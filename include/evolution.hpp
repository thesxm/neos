#ifndef NEOS_INCLUDE_EVOLUTION_H
#define NEOS_INCLUDE_EVOLUTION_H

#include "./neural.hpp"

using NEURAL::Network;

namespace EVOLUTION {
    vector<float> generate_random_weights(const int &n);
    vector<Network> generate_initial_population(const int &population_size, const int &input_length, const int &output_length, const vector<int> &neuron_counts);
    // TODO: Create the neuroevolve() function
};

#endif