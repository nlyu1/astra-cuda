#include "astra_utils.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <initializer_list>
#include <list>

namespace astra {

[[noreturn]] void AstraFatalError(const std::string& error_msg) {
    std::cerr << "Astra Fatal Error: " << error_msg << std::endl
            << std::endl
            << std::flush;
    throw std::runtime_error("Astra Fatal Error: " + error_msg);
}

// String replacement function
std::string StrReplaceAll(const std::string& input, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return input;
    }
    
    std::string result = input;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

// String replacement function (initializer list version)
std::string StrReplaceAll(const std::string& input, std::initializer_list<std::pair<std::string, std::string>> replacements) {
    std::string result = input;
    for (const auto& replacement : replacements) {
        result = StrReplaceAll(result, replacement.first, replacement.second);
    }
    return result;
}

// String splitting function (with string delimiter)
std::vector<std::string> StrSplit(const std::string& input, const std::string& delimiter) {
    std::vector<std::string> result;
    if (input.empty()) {
        return result;
    }
    
    if (delimiter.empty()) {
        result.push_back(input);
        return result;
    }
    
    size_t start = 0;
    size_t pos = 0;
    while ((pos = input.find(delimiter, start)) != std::string::npos) {
        result.push_back(input.substr(start, pos - start));
        start = pos + delimiter.length();
    }
    result.push_back(input.substr(start));
    return result;
}

// String splitting function (with char delimiter)
std::vector<std::string> StrSplit(const std::string& input, char delimiter) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        result.push_back(token);
    }
    return result;
}

// Helper function to split string with max splits limit
std::vector<std::string> StrSplitWithLimit(const std::string& input, const std::string& delimiter, int max_splits) {
    std::vector<std::string> result;
    if (input.empty()) {
      return result;
    }
    
    if (delimiter.empty()) {
      result.push_back(input);
      return result;
    }
    
    size_t start = 0;
    size_t pos = 0;
    int splits = 0;
    
    while ((pos = input.find(delimiter, start)) != std::string::npos && splits < max_splits) {
      result.push_back(input.substr(start, pos - start));
      start = pos + delimiter.length();
      splits++;
    }
    // Add the remaining part
    result.push_back(input.substr(start));
    return result;
  }
  

// Split into exactly two parts on first occurrence of delimiter
std::pair<std::string, std::string> StrSplitFirst(const std::string& input, const std::string& delimiter) {
    size_t pos = input.find(delimiter);
    if (pos == std::string::npos) {
        return std::make_pair(input, "");
    }
    return std::make_pair(input.substr(0, pos), input.substr(pos + delimiter.length()));
}

// String to int conversion with error handling
bool SimpleAtoi(const std::string& str, int* value) {
    try {
        size_t pos;
        int result = std::stoi(str, &pos);
        if (pos != str.length()) {
            return false; // Not all characters were converted
        }
        *value = result;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// String to double conversion with error handling
bool SimpleAtod(const std::string& str, double* value) {
    try {
        size_t pos;
        double result = std::stod(str, &pos);
        if (pos != str.length()) {
            return false; // Not all characters were converted
        }
        *value = result;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Format double with appropriate precision
std::string FormatDouble(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(15) << value;
    std::string result = oss.str();
    
    // Remove trailing zeros after decimal point
    if (result.find('.') != std::string::npos) {
        while (result.length() > 1 && result.back() == '0') {
            result.pop_back();
        }
        // If we removed all digits after decimal, add one zero
        if (result.back() == '.') {
            result += '0';
        }
    } else {
        // Add .0 for integers to make it clear it's a double
        result += ".0";
    }
    
    return result;
}

}