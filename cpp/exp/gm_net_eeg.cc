#include <string>
#include "cpp/networks/controller.cc"
#include "cpp/networks/gm_network.cc"


using namespace controller;
using namespace gm_network;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    controller::Controller<gm_network::GmNet> ctrl(cfg);
    ctrl.InitializeSimulation();
    ctrl.PrintStarNets();
//    ctrl.TrainNetworks();

    return 0;
}