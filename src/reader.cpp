#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "../include/reader.hpp"

using std::cerr;
using std::endl;
using std::ifstream;
using std::getline;
using std::string;
using std::vector;
using std::runtime_error;
using std::stof;

namespace READER
{
    vector<vector<vector<float> *> *> *read_from_file(string file_name)
    {
        ifstream i_file(file_name);

        if (!i_file) {
            cerr << "Failed to open file `" << file_name << "`." << endl;

            exit(0);
        }

        vector<vector<float> *> *inps = new vector<vector<float> *>;
        vector<vector<float> *> *outs = new vector<vector<float> *>;

        string line;
        while (getline(i_file, line)) {
            int _ = line.find(';');

            string inputs_string = line.substr(0, _);
            string outputs_string = line.substr(_ + 1);

            vector<float> *inp = new vector<float>;
            vector<float> *out = new vector<float>;

            int i = 0;
            string f = "";
            while (i <= inputs_string.size()) {
                if (i == inputs_string.size() || inputs_string[i] == ',') {
                    inp->push_back(stof(f));
                    f = "";
                } else {
                    f += inputs_string[i];
                }

                i++;
            }

            i = 0;
            f = "";
            while (i <= outputs_string.size()) {
                if (i == outputs_string.size() || outputs_string[i] == ',') {
                    out->push_back(stof(f));
                    f = "";
                } else {
                    f += outputs_string[i];
                }
                
                i++;
            }

            inps->push_back(inp);
            outs->push_back(inp);
        }

        return new vector<vector<vector<float> *> *>({inps, outs});
    }
}