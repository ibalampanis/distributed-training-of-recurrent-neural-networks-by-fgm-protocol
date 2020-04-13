#include <string>
#include "cpp/networks/controller.cc"
#include "cpp/networks/fgm.hh"


using namespace controller;
using namespace fgm;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    controller::Controller<FgmNet> ctrl(cfg);
    ctrl.InitializeSimulation();
    ctrl.ShowNetworkInfo();
//    ctrl.TrainNetworks();

    return 0;
}