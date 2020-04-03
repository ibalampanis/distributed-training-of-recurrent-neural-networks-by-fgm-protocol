#include <string>
#include "cpp/networks/controller.cc"
#include "cpp/networks/gm_network.cc"

using namespace std;
using namespace controller;
using namespace gm_network;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    controller::Controller<gm_network::GmNet> ctrl(cfg);
    ctrl.InitializeSimulation();
    ctrl.ShowNetworkInfo();
//    ctrl.TrainNetworks();

    return 0;
}