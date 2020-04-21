#include <string>
#include "cpp/networks/supervisor.cc"
#include "cpp/networks/gm.cc"

using namespace std;
using namespace supervisor;
using namespace algorithms::gm;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    Supervisor<GmNet> sv(cfg);
    sv.InitializeSimulation();
    sv.ShowNetworkInfo();
    sv.TrainOverNetwork();

    return 0;
}