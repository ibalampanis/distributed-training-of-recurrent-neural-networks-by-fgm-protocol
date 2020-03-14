#include <string>
#include "cpp/networks/controller.hh"
#include "cpp/networks/gm_network.hh"


using namespace controller;
using namespace gm_network;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    Controller<gm_network::GmNet> sim(cfg);
    sim.InitializeSimulation();
    sim.PrintStarNets();
//    sim.TrainNetworks();

    return 0;
}