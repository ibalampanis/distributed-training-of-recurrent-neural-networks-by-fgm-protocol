#include <string>
#include "cpp/networks/control.cc"
#include "cpp/networks/fgm.cc"


using namespace controller;
using namespace algorithms::fgm;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    Controller<FgmNet> sv(cfg);
    sv.InitializeSimulation();
//    sv.ShowNetworkInfo();
    sv.TrainOverNetwork();

    return 0;
}