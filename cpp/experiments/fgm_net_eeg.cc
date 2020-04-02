#include <string>
#include "cpp/networks/controller.cc"
#include "cpp/networks/fgm_network.cc"


using namespace controller;
using namespace fgm_network;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    controller::Controller<fgm_network::FgmNet> ctrl(cfg);
    ctrl.InitializeSimulation();
    ctrl.PrintNetInfo();
//    ctrl.TrainNetworks();

    return 0;
}