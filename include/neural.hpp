#include <vector>

#ifndef NEOS_INCLUDE_NEURAL_H
#define NEOS_INCLUDE_NEURAL_H

using std::vector;

namespace NEURAL {
    enum NEURAL_T {
        NEURON_T,
        LAYER_T,
        NETWORK_T,
    };

    enum NEURON_ACTIVATION_FUNCTION_T {
        RELU,
        SIGMOID,
        TANH,
    };

    float apply_activation_function(float &x, NEURON_ACTIVATION_FUNCTION_T &t);

    class Base {
        protected:
        NEURAL_T _t;

        public:
        Base(NEURAL_T type);
        NEURAL_T type();
    };

    class Neuron: public Base {
        private:
        vector<float> _w;
        NEURON_ACTIVATION_FUNCTION_T _fn;

        public:
        Neuron(vector<float> &w, NEURON_ACTIVATION_FUNCTION_T &fn);
        int weight_count();
        float operator<<(vector<float> &inp);
        friend Neuron operator*(Neuron &a, Neuron &b);
    };

    class Layer: public Base {
        private:
        vector<Neuron> _n;

        public:
        Layer(vector<vector<float>> &w, vector<NEURON_ACTIVATION_FUNCTION_T> &fn);
        Layer(vector<Neuron> &w);
        int neuron_count();
        vector<float> operator<<(vector<float> &inp);
        friend Layer operator*(Layer &a, Layer &b);
    };

    class Network: public Base {
        private:
        vector<Layer> _l;

        public:
        Network(vector<vector<vector<float>>> &w, vector<vector<NEURON_ACTIVATION_FUNCTION_T>> &fn);
        Network(vector<Layer> &w);
        int layer_count();
        vector<float> operator<<(vector<float> &inp);
        friend Network operator*(Network &a, Network &b);
    };
};

#endif