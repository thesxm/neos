#include <cmath>
#include <ctime>
#include "../include/evolution.hpp"
#include "../include/neural.hpp"

using NEURAL::Network;
using NEURAL::Layer;
using NEURAL::Neuron;
using NEURAL::NEURON_ACTIVATION_FUNCTION_T;

namespace EVOLUTION {
    vector<float> generate_random_weights(const int &n) {
        vector<float> res;

        for (int i = 0; i < n; i++) {
            res.push_back(1.0 * rand() / RAND_MAX);
        }

        return res;
    }

    vector<Network> generate_initial_population(const int &population_size, const int &input_length, const int &output_length, const vector<int> &neuron_counts) {
        vector<Network> res;
        
        for (int i = 0; i < population_size; i++) {
            vector<Layer> layers;
            int inputs_per_neuron = input_length;
            for (int j: neuron_counts) {
                vector<Neuron> neurons;
                for (int k = 0; k < j; k++) {
                    vector<float> w = generate_random_weights(inputs_per_neuron);
                    NEURON_ACTIVATION_FUNCTION_T fn = NEURON_ACTIVATION_FUNCTION_T::RELU;
                    neurons.push_back(Neuron(w, fn));
                }

                layers.push_back(Layer(neurons));
                inputs_per_neuron = j;
            }

            vector<Neuron> output_neurons;
            for (int j = 0; j < output_length; j++) {
                vector<float> w = generate_random_weights(inputs_per_neuron);
                NEURON_ACTIVATION_FUNCTION_T fn = NEURON_ACTIVATION_FUNCTION_T::TANH;
                output_neurons.push_back(Neuron(w, fn));
            }
            
            layers.push_back(Layer(output_neurons));

            res.push_back(
                Network(layers)
            );
        }

        return res;
    }
};