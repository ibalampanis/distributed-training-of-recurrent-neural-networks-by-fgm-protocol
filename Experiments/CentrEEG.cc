#include "rnn_models/predictor/rnn_learner.hh"
#include <string>


using namespace rnn_learner;

int main(int argc, char **argv) {

    std::string cfg = std::string(argv[1]);

    auto *pLearner = new RNNLearner(cfg);

    pLearner->CentralizedDataPreparation();
    pLearner->TrainModel();
    pLearner->MakePrediction();

    return 0;
}


