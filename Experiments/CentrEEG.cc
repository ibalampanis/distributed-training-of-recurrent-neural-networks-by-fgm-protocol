#include "RNN_models/predictor/RNNPredictor.hh"
#include <string>


using namespace rnn_predictor;

int main(int argc, char **argv) {

    std::string cfg = std::string(argv[1]);

    auto *predictor = new RNNPredictor(cfg, "gm_net");

    predictor->CentralizedDataPreparation();
    predictor->TrainModel();
    predictor->MakePrediction();

    return 0;
}


