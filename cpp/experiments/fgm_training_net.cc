#include <string>
#include "cpp/networks/controller.cc"
#include "cpp/networks/fgm.cc"


using namespace controller;
using namespace algorithms::fgm;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    controller::Controller<FgmNet> ctrl(cfg);
    ctrl.InitializeSimulation();
    ctrl.ShowNetworkInfo();
    ctrl.TrainOverNetwork();

    return 0;
}