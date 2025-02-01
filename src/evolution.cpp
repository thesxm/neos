#include <cmath>
#include <ctime>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include "../include/evolution.hpp"
#include "../include/neural.hpp"

#include <iostream>

using std::find;
using std::priority_queue;
using std::unordered_map;
using std::unordered_set;

using NEURAL::Layer;
using NEURAL::Network;
using NEURAL::Neuron;
using NEURAL::NEURON_ACTIVATION_FUNCTION_T;

namespace EVOLUTION
{
    vector<float> *generate_random_weights(const int &n)
    {
        vector<float> *res = new vector<float>;

        for (int i = 0; i < n; i++)
        {
            res->push_back(-1 + 2.0 * rand() / RAND_MAX);
        }

        return res;
    }

    PQUEUE_NODE::PQUEUE_NODE(Network *network, const float &fitness_score) : network(network), fitness_score(fitness_score) {};

    PQUEUE_NODE::~PQUEUE_NODE()
    {
        delete this->network;
    }

    bool PQUEUE_COMPARE::operator()(PQUEUE_NODE *a, PQUEUE_NODE *b)
    {
        return a->fitness_score < b->fitness_score;
    }

    vector<Network *> *Environment::generate_initial_population()
    {
        vector<Network *> *res = new vector<Network *>;

        for (int i = 0; i < this->population_size; i++)
        {
            vector<Layer *> layers;
            int inputs_per_neuron = this->input_length;
            for (int j : *(this->neuron_counts))
            {
                vector<Neuron *> neurons;
                for (int k = 0; k < j; k++)
                {
                    vector<float> *w = generate_random_weights(inputs_per_neuron);
                    NEURON_ACTIVATION_FUNCTION_T fn = NEURON_ACTIVATION_FUNCTION_T::RELU;
                    neurons.push_back(new Neuron(*w, fn));
                }

                layers.push_back(new Layer(neurons));
                inputs_per_neuron = j;
            }

            vector<Neuron *> output_neurons;
            for (int j = 0; j < this->output_length; j++)
            {
                vector<float> *w = generate_random_weights(inputs_per_neuron);
                NEURON_ACTIVATION_FUNCTION_T fn = NEURON_ACTIVATION_FUNCTION_T::NO_ACTIVATION;
                output_neurons.push_back(new Neuron(*w, fn));
            }

            layers.push_back(new Layer(output_neurons));

            res->push_back(
                new Network(layers));
        }

        return res;
    }

    Environment::Environment(
        const int &population_size,
        const float &selection_ratio,
        const float &mutation_ratio,
        const int &input_length,
        const int &output_length,
        vector<int> *neuron_counts,
        float (*fitness_function)(vector<float> *network_output, vector<float> *ideal_output)) : population_size(population_size), selection_ratio(selection_ratio), mutation_ratio(mutation_ratio), input_length(input_length), output_length(output_length), neuron_counts(neuron_counts), fitness_function(fitness_function)
    {
        this->population = this->generate_initial_population();
    }

    vector<Network *> *Environment::evolve_one_generation(vector<vector<float> *> *inps, vector<vector<float> *> *outs)
    {
        priority_queue<PQUEUE_NODE *, vector<PQUEUE_NODE *>, PQUEUE_COMPARE> pq_max;
        priority_queue<PQUEUE_NODE *, vector<PQUEUE_NODE *>, PQUEUE_COMPARE> pq_min;

        for (auto n : *(this->population))
        {
            float fitness_score = 0;
            for (int i = 0; i < inps->size(); i++)
            {
                vector<float> *inp = inps->at(i);
                vector<float> *network_output = *n << *inp;
                float t = this->fitness_function(network_output, inp);

                fitness_score += pow(t, 2);
            }

            fitness_score = sqrt(fitness_score);

            std::cout << fitness_score << std::endl;

            pq_max.push(new PQUEUE_NODE(
                n,
                fitness_score));

            pq_min.push(new PQUEUE_NODE(
                n,
                100 - fitness_score));
        }

        vector<Network *> parents;
        int top_networks_count = .9 * this->selection_ratio * this->population_size;
        int bottom_networks_count = .1 * this->selection_ratio * this->population_size;

        for (int i = 0; i < top_networks_count; i++)
        {
            parents.push_back(pq_max.top()->network);
            pq_max.pop();
        }

        for (int i = 0; i < bottom_networks_count; i++)
        {
            parents.push_back(pq_min.top()->network);
            pq_min.pop();
        }

        while (!pq_max.empty())
        {
            PQUEUE_NODE *p = pq_max.top();
            pq_max.pop();

            if (find(parents.begin(), parents.end(), p->network) == parents.end())
                delete p;
        }

        vector<Network *> children;
        for (int i = 0; i < (this->population_size - parents.size()); i++)
        {
            int parent_a_index, parent_b_index;

            do
            {
                parent_a_index = rand() % parents.size();
                parent_b_index = rand() % parents.size();
            } while (parent_a_index == parent_b_index);

            Network *parent_a = parents[parent_a_index];
            Network *parent_b = parents[parent_b_index];
            Network *child = *parent_a * *parent_b;

            children.push_back(child);
        }

        vector<Network *> *res = new vector<Network *>(parents.begin(), parents.end());
        res->insert(res->end(), children.begin(), children.end());

        unordered_set<int> mutated_networks;
        for (int i = 0; i < (int)res->size() * this->mutation_ratio; i++)
        {
            int network_index;

            do
            {
                network_index = rand() % res->size();
            } while (mutated_networks.find(network_index) != mutated_networks.end());

            res->at(network_index)->mutate();
            mutated_networks.insert(network_index);
        }

        return res;
    }

    vector<Network *> *Environment::evolve_n_generations(int n, vector<vector<float> *> *inps, vector<vector<float> *> *outs)
    {
        for (int i = 0; i < n; i++)
        {
            this->population = this->evolve_one_generation(
                inps,
                outs);
        }

        return this->population;
    }
};