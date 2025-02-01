#include <vector>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <iostream>
#include "../include/neural.hpp"

using std::exp;
using std::max;
using std::min;
using std::ostream;
using std::runtime_error;
using std::vector;

namespace NEURAL
{
    float apply_activation_function(float &x, NEURON_ACTIVATION_FUNCTION_T &t)
    {
        switch (t)
        {
        case NEURON_ACTIVATION_FUNCTION_T::RELU:
            return x >= 0 ? x : 0;
        case NEURON_ACTIVATION_FUNCTION_T::SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case NEURON_ACTIVATION_FUNCTION_T::TANH:
            return tanh(x);
        case NEURON_ACTIVATION_FUNCTION_T::NO_ACTIVATION:
            return x;
        default:
            throw runtime_error("Unknown activation function received.");
            exit(1);
        }
    }

    Base::Base(NEURAL_T type)
    {
        this->_t = type;
    };

    NEURAL_T Base::type()
    {
        return this->_t;
    }

    Neuron::Neuron(vector<float> &w, NEURON_ACTIVATION_FUNCTION_T &fn) : Base(NEURAL_T::NEURON_T), _fn(fn)
    {
        this->_w = vector<float>(w.begin(), w.end());
    };

    int Neuron::weight_count()
    {
        return this->_w.size();
    }

    void Neuron::mutate()
    {
        int n = this->weight_count();
        for (int i = 0; i < n / 2; i++)
        {
            // this->_w[i] = this->_w[i] * (-3 + 6.0 * rand() / RAND_MAX);
            std::swap(this->_w[i], this->_w[n - i - 1]);
        }
    }

    float Neuron::operator<<(const vector<float> &inp)
    {
        int n = this->weight_count();
        float res = 0;

        for (int i = 0; i < n; i++)
            res += this->_w[i] * inp[i];

        return apply_activation_function(res, this->_fn);
    }

    Neuron *operator*(Neuron &a, Neuron &b)
    {
        int n = a.weight_count();
        vector<float> w;

        // for (int i = 0; i < n; i++) {
        //     float gap = abs(a._w[i] - b._w[i]);
        //     float low = min(a._w[i], b._w[i]) - gap;
        //     float high = max(a._w[i], b._w[i]) + gap;

        //     w.push_back(
        //         low + 3 * gap * rand() / RAND_MAX
        //     );
        // }

        for (int i = 0; i < n; i++)
        {
            w.push_back(
                (1.0 * rand() / RAND_MAX) <= .5 ? a._w[i] : b._w[i]);
        }

        return new Neuron(w, a._fn);
    }

    ostream &operator<<(ostream &o, Neuron &n)
    {
        o << "[";
        for (auto i : n._w)
            o << " " << i;
        o << " ]";

        return o;
    }

    Layer::Layer(vector<vector<float>> &w, vector<NEURON_ACTIVATION_FUNCTION_T> &fn) : Base(NEURAL_T::LAYER_T)
    {
        this->_n = vector<Neuron *>();

        int n = w.size();
        for (int i = 0; i < n; i++)
        {
            this->_n.push_back(
                new Neuron(w[i], fn[i]));
        }
    };

    Layer::Layer(vector<Neuron *> &w) : Base(NEURAL_T::LAYER_T)
    {
        this->_n = vector<Neuron *>(w.begin(), w.end());
    }

    Layer::~Layer()
    {
        for (auto n : this->_n)
            delete n;
    }

    int Layer::neuron_count()
    {
        return this->_n.size();
    }

    void Layer::mutate()
    {
        for (auto n : this->_n)
            n->mutate();
    }

    vector<float> *Layer::operator<<(const vector<float> &inp)
    {
        int n = this->neuron_count();
        vector<float> *res = new vector<float>;

        for (int i = 0; i < n; i++)
            res->push_back(*this->_n[i] << inp);

        return res;
    }

    Layer *operator*(Layer &a, Layer &b)
    {
        int n = a.neuron_count();
        vector<Neuron *> w;

        for (int i = 0; i < n; i++)
            w.push_back(*a._n[i] * *b._n[i]);

        return new Layer(w);
    }

    ostream &operator<<(ostream &o, Layer &l)
    {
        o << "[";
        for (auto _ : l._n)
            o << *_;
        o << "]";

        return o;
    }

    Network::Network(vector<vector<vector<float>>> &w, vector<vector<NEURON_ACTIVATION_FUNCTION_T>> &fn) : Base(NEURAL_T::NETWORK_T)
    {
        this->_l = vector<Layer *>();

        int n = w.size();
        for (int i = 0; i < n; i++)
        {
            this->_l.push_back(
                new Layer(w[i], fn[i]));
        }
    }

    Network::Network(vector<Layer *> &w) : Base(NEURAL_T::NETWORK_T)
    {
        this->_l = vector<Layer *>(w.begin(), w.end());
    }

    Network::~Network()
    {
        for (auto l : this->_l)
            delete l;
    }

    int Network::layer_count()
    {
        return this->_l.size();
    }

    void Network::mutate()
    {
        for (auto l : this->_l)
            l->mutate();
    }

    vector<float> *Network::operator<<(const vector<float> &inp)
    {
        int n = this->layer_count();
        vector<float> *res = new vector<float>(inp.begin(), inp.end());

        for (int i = 0; i < n; i++)
            res = *this->_l[i] << *res;

        return res;
    }

    Network *operator*(Network &a, Network &b)
    {
        int n = a.layer_count();
        vector<Layer *> w;

        for (int i = 0; i < n; i++)
            w.push_back(*a._l[i] * *b._l[i]);

        return new Network(w);
    }

    ostream &operator<<(ostream &o, Network &n)
    {
        o << "[";
        for (auto _ : n._l)
            o << *_;
        o << "]";

        return o;
    }
};