#include <string>
#include "cpp/networks/supervisor.cc"
#include "cpp/networks/fgm.cc"


using namespace supervisor;
using namespace algorithms::fgm;
using namespace std;

int main(int argc, char **argv) {

    string cfg = string(argv[1]);

    Supervisor<FgmNet> sv(cfg);
    sv.InitializeSimulation();
    sv.ShowNetworkInfo();
    sv.TrainOverNetwork();

    return 0;
}