#include "Models/rnn_learner.hh"
#include <string>


using namespace rnn_learner;
using namespace std;

int main(int argc, char **argv) {

    if (argc != 2) return -1;

    string cfg = string(argv[1]);

    auto *pLearner = new RNNLearner(cfg);

    pLearner->CentralizedDataPreparation();
    pLearner->TrainModel();
    pLearner->MakePrediction();

    return 0;
}


