#include "RNN_models/predictor/RNNPredictor.hh"

using namespace rnn_predictor;

int main() {

    /* Constructor arguments:
    **      1. Training epochs
    **      2. LSTM cells
    **      3. Rho
    **      4. SGD step size
    **      5. SGD batch size
    **      6. SGD Max iterations
    */
    auto *predictor = new RNNPredictor(200, 10, 15, 5e-5, 16, 3000);
//    auto *predictor = new RNNPredictor(20, 30, 25, 4.5e-5, 128, 10000);


    predictor->DataPreparation();
    predictor->TrainModel();
    predictor->MakePrediction();

    return 0;
}


