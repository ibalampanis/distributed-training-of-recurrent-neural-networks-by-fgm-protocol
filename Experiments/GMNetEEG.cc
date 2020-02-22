#include <string>
#include "Networks/feeders.hh"


using namespace feeders;

int main(int argc, char **argv) {

    std::string cfg = std::string(argv[1]);

    Random_Feeder<gm_protocol::GM_Net> simulation(cfg);

    simulation.initializeSimulation();
    simulation.printStarNets();
//    simulation.TrainNetworks();

    return 0;
}