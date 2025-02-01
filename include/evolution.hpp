#ifndef NEOS_INCLUDE_EVOLUTION_H
#define NEOS_INCLUDE_EVOLUTION_H

#include "./neural.hpp"

using NEURAL::Network;

namespace EVOLUTION
{
    vector<float> *generate_random_weights(const int &n);

    struct PQUEUE_NODE
    {
        Network *network;
        float fitness_score;
        PQUEUE_NODE(Network *network, const float &fitness_score);
        ~PQUEUE_NODE();
    };

    struct PQUEUE_COMPARE
    {
        bool operator()(PQUEUE_NODE *a, PQUEUE_NODE *b);
    };

    class Environment
    {
    private:
        int population_size;
        float selection_ratio;
        float mutation_ratio;
        int input_length;
        int output_length;
        vector<int> *neuron_counts;
        float (*fitness_function)(vector<float> *network_output, vector<float> *ideal_output);
        vector<Network *> *population;
        vector<Network *> *generate_initial_population();

    public:
        Environment(
            const int &population_size,
            const float &selection_ratio,
            const float &mutation_ratio,
            const int &input_length,
            const int &output_length,
            vector<int> *neuron_counts,
            float (*fitness_function)(vector<float> *network_output, vector<float> *ideal_output));
        vector<Network *> *evolve_one_generation(vector<vector<float> *> *inps, vector<vector<float> *> *outs);
        vector<Network *> *evolve_n_generations(int n, vector<vector<float> *> *inps, vector<vector<float> *> *outs);
        // vector<Network*> *evolve_until(const vector<float> &inp, const vector<float> &out, bool (*evolve_condition)(const vector<float> &fitness_scores));
    };
};

#endif