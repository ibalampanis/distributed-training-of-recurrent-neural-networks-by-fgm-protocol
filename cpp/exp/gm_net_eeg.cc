#include <string>
#include "cpp/networks/controller.hh"

using namespace controller;
using namespace gm_protocol;

int main(int argc, char **argv) {

    std::string cfg = std::string(argv[1]);

    Controller<GmNet> sim(cfg);
    sim.InitializeSimulation(); // TODO
    sim.PrintStarNets();
    sim.TrainNetworks();

    return 0;
}