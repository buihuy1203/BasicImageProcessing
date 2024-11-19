#include "readfile.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::string loadKernelSourceFile(const std::string &filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}
