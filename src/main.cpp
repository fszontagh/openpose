#include "openpose.hpp"
#include <iostream>

int main(int argc, char **argv) {
    // A try-catch blokk elkapja a modell betöltésekor vagy a feldolgozás során
    // keletkező kritikus hibákat, és tiszta hibaüzenettel áll le.
    try {
        OpenPose openpose(argc, argv);
        openpose.run();
    } catch (const std::exception& e) {
        std::cerr << "An unrecoverable error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

