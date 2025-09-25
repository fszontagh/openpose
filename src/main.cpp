#include "openpose.hpp"
#include "parameter_handler.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        ParameterHandler param_handler;
        ProgramOptions opts = param_handler.parse(argc, argv);

        OpenPose openpose(opts);
        openpose.run();
    } catch (const ParameterException& e) {
        std::cerr << "Error parsing parameters: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "An unrecoverable error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

