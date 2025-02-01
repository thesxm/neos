#ifndef NEOS_INCLUDE_READER_H
#define NEOS_INCLUDE_READER_h

#include <vector>
#include <string>

using std::string;
using std::vector;

namespace READER {
    vector<vector<vector<float> *> *> *read_from_file(string file_name);
}

#endif