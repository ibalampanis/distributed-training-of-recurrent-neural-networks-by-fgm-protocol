#include <string>
#include "Networks/feeders.hh"

using namespace feeders;
using namespace gm_protocol;

int main(int argc, char **argv) {

    std::string cfg = std::string(argv[1]);

    Random_Feeder<GM_Net> sim(cfg);
    sim.initializeSimulation();
    sim.printStarNets();
//    sim.TrainNetworks();

    return 0;
}