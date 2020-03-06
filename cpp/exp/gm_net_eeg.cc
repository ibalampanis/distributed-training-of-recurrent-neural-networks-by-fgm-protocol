#include <string>
#include "cpp/networks/feeders.hh"

using namespace feeders;
using namespace gm_protocol;

int main(int argc, char **argv) {

    std::string cfg = std::string(argv[1]);

    RandomFeeder<GM_Net> sim(cfg);
    sim.InitializeSimulation(); // TODO
    sim.PrintStarNets();
//    sim.TrainNetworks();

    return 0;
}