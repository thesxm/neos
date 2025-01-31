#include <vector>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include "../include/neural.hpp"

using std::vector;
using std::max;
using std::exp;
using std::runtime_error;

namespace NEURAL {
    float apply_activation_function(float &x, NEURON_ACTIVATION_FUNCTION_T &t) {
        switch (t) {
            case NEURON_ACTIVATION_FUNCTION_T::RELU: return x >= 0 ? x : 0;
            case NEURON_ACTIVATION_FUNCTION_T::SIGMOID: return 1.0 / (1.0 + exp(-x));
            case NEURON_ACTIVATION_FUNCTION_T::TANH: return tanh(x);
            default: throw runtime_error("Unknown activation function received."); exit(1);
        }
    }

    Base::Base(NEURAL_T type) {
        this->_t = type;
    };

    NEURAL_T Base::type() {
        return this->_t;
    }

    Neuron::Neuron(vector<float> &w, NEURON_ACTIVATION_FUNCTION_T &fn): Base(NEURAL_T::NEURON_T), _fn(fn) {
        this->_w = vector<float>(w.begin(), w.end());
    };

    int Neuron::weight_count() {
        return this->_w.size();
    }

    float Neuron::operator<<(vector<float> &inp) {
        int n = this->weight_count();
        float res = 0;

        for (int i = 0; i < n; i++) res += this->_w[i] * inp[i];

        return apply_activation_function(res, this->_fn);
    }

    Neuron operator*(Neuron &a, Neuron &b) {
        int n = a.weight_count();
        vector<float> w;

        float t = 1.0 * rand() / RAND_MAX;
        if (t <= 0.33) for (int i = 0; i < n; i++) w.push_back(2 * a._w[i] - (a._w[i] + b._w[i]) / 2);
        else if (t <= 0.66) for (int i = 0; i < n; i++) w.push_back((a._w[i] + b._w[i]) / 2);
        else for (int i = 0; i < n; i++) w.push_back(2 * b._w[i] - (a._w[i] + b._w[i]) / 2);

        return Neuron(w, a._fn);
    }

    Layer::Layer(vector<vector<float>> &w, vector<NEURON_ACTIVATION_FUNCTION_T> &fn): Base(NEURAL_T::LAYER_T) {
        this->_n = vector<Neuron>();

        int n = w.size();
        for (int i = 0; i < n; i++) {
            this->_n.push_back(
                Neuron(w[i], fn[i])
            );
        }
    };

    Layer::Layer(vector<Neuron> &w): Base(NEURAL_T::LAYER_T) {
        this->_n = vector<Neuron>(w.begin(), w.end());
    }

    int Layer::neuron_count() {
        return this->_n.size();
    }

    vector<float> Layer::operator<<(vector<float> &inp) {
        int n = this->neuron_count();
        vector<float> res = vector<float>();

        for (int i = 0; i < n; i++) res.push_back(this->_n[i] << inp);

        return res;
    }

    Layer operator*(Layer &a, Layer &b) {
        int n = a.neuron_count();
        vector<Neuron> w;

        for (int i = 0; i < n; i++) w.push_back(a._n[i] * b._n[i]);

        return Layer(w);
    }

    Network::Network(vector<vector<vector<float>>> &w, vector<vector<NEURON_ACTIVATION_FUNCTION_T>> &fn): Base(NEURAL_T::NETWORK_T) {
        this->_l = vector<Layer>();

        int n = w.size();
        for (int i = 0; i < n; i++) {
            this->_l.push_back(
                Layer(w[i], fn[i])
            );
        }
    }

    Network::Network(vector<Layer> &w): Base(NEURAL_T::NETWORK_T) {
        this->_l = vector<Layer>(w.begin(), w.end());
    }

    int Network::layer_count() {
        return this->_l.size();
    }

    vector<float> Network::operator<<(vector<float> &inp) {
        int n = this->layer_count();
        vector<float> res = vector<float>(inp.begin(), inp.end());

        for (int i = 0; i < n; i++) res = this->_l[i] << res;

        return res;
    }

    Network operator*(Network &a, Network &b) {
        int n = a.layer_count();
        vector<Layer> w;

        for (int i = 0; i < n; i++) w.push_back(a._l[i] * b._l[i]);

        return Network(w);
    }
};