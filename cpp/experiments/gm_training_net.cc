#include <string>
#include "cpp/networks/control.cc"
#include "cpp/networks/gm.cc"

using namespace std;
using namespace controller;
using namespace algorithms::gm;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    Controller<GmNet> sv(cfg);
    sv.InitializeSimulation();
    sv.ShowNetworkInfo();
    sv.TrainOverNetwork();

    return 0;
}