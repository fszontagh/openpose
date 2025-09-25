#ifndef PARAMETER_HANDLER_HPP
#define PARAMETER_HANDLER_HPP

#include <string>
#include <vector>
#include <stdexcept>

// Struct to hold all configurable options
struct ProgramOptions {
    std::string image_path;
    std::string output_file;
    float blend_factor = 0.5f;
    float threshold = 0.1f;
    float person_threshold = 0.5f;
    float nms_threshold = 0.4f;
    std::string models_dir = "./models";
    bool enable_validation = true;
    bool enable_interpolation = true;
    bool process_body = true;
    bool process_hand = false;
    bool process_face = false;
    bool verbose = false;
    bool multi_person = true; // Changed default to true
    bool draw_foot = false;
    bool no_rainbow = false;
};

// Custom exception for parsing errors
class ParameterException : public std::runtime_error {
public:
    explicit ParameterException(const std::string& message) : std::runtime_error(message) {}
};

// Class to handle command-line argument parsing and validation
class ParameterHandler {
public:
    ParameterHandler() = default;
    ProgramOptions parse(int argc, char** argv);

private:
    void displayHelp();
    float parseFloat(const std::string& s, const std::string& arg_name);
};

#endif // PARAMETER_HANDLER_HPP