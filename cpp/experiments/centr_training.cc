#include "cpp/models/rnn.hh"
#include <string>


using namespace rnn;
using namespace std;

int main(int argc, char **argv) {

    if (argc != 2) return -1;

    string cfg = string(argv[1]);

    auto *pLearner = new RnnLearner(cfg, RNN<MeanSquaredError<>, HeInitialization>(0));

    pLearner->CentralizedDataPreparation();
    pLearner->BuildModel();
    pLearner->TrainModel();
    pLearner->MakePrediction();

    return 0;
}


