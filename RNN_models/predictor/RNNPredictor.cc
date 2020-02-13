#include "RNNPredictor.hh"
#include <armadillo>

using namespace rnn_predictor;
using namespace arma;

RNNPredictor::RNNPredictor(int trainingEpochs, int lstmCells, int rho, double stepSize, int batchSize,
                           int iterationsPerEpoch) : trainingEpochs(trainingEpochs), lstmCells(lstmCells), rho(rho),
                                                     stepSize(stepSize), batchSize(batchSize),
                                                     iterationsPerEpoch(iterationsPerEpoch) {}

//template<typename InputDataType = arma::mat, typename DataType = arma::cube, typename LabelType = arma::cube>
void RNNPredictor::CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, const size_t rho) {
    for (size_t i = 0; i < dataset.n_cols - rho; i++) {
        X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(), arma::span(i, i + rho - 1));
        y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(
                arma::span(dataset.n_rows - 1, dataset.n_rows - 1), arma::span(i + 1, i + rho));
    }
}

double RNNPredictor::MSECalc(arma::cube &pred, arma::cube &Y) {
    double err_sum = 0.0;
    arma::cube diff = pred - Y;
    for (size_t i = 0; i < diff.n_slices; i++) {
        arma::mat temp = diff.slice(i);
        err_sum += accu(temp % temp);
    }
    return (err_sum / (diff.n_elem + 1e-50));
}

double RNNPredictor::TakeVectorAVG(const std::vector<double> &vec) {
    double sum = 0;
    for (double i : vec)
        sum += i;

    double avg = sum / vec.size();
    return avg;
}

void RNNPredictor::DataPreparation() {

    arma::mat dataset;
    // In Armadillo rows represent features, columns represent data points.
    cout << "Reading dataset ...";
    data::Load(dataFile, dataset, true);
    cout << " OK." << endl;

    // Scale all data into the range (0, 1) for increased numerical stability.
    data::MinMaxScaler scale;
    scale.Fit(dataset);
    scale.Transform(dataset, dataset);


    // We need to represent the input data for RNN in an arma::cube (3D matrix).
    // The 3rd dimension is rho, the number of past data records the RNN uses for learning.
    arma::cube X, y;
    X.set_size(inputSize, dataset.n_cols - rho + 1, rho);
    y.set_size(outputSize, dataset.n_cols - rho + 1, rho);


    CreateTimeSeriesData(dataset, X, y, rho);

    // Split the data into training and testing sets.
    size_t trainingSize = (1 - trainTestRatio) * X.n_cols;
    trainX = X.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
    trainY = y.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
    testX = X.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());
    testY = y.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());

}

void RNNPredictor::TrainModel() {

    // Model definition
    RNN<MeanSquaredError<>, HeInitialization> model(rho);

    // Model building
    model.Add<IdentityLayer<> >();
    model.Add<LSTM<> >(inputSize, lstmCells, maxRho);
    model.Add<Dropout<> >(0.5);
    model.Add<ReLULayer<> >();
    model.Add<LSTM<> >(lstmCells, lstmCells, maxRho);
    model.Add<Dropout<> >(0.5);
    model.Add<ReLULayer<> >();
    model.Add<LSTM<> >(lstmCells, lstmCells, maxRho);
    model.Add<ReLULayer<> >();
    model.Add<Linear<> >(lstmCells, outputSize);


    // Define and set parameters for the Stochastic Gradient Descent (SGD) optimizer.
    SGD<AdamUpdate> optimizer(stepSize, batchSize, iterationsPerEpoch, tolerance,
                              bShuffle, AdamUpdate(epsilon, beta1, beta2));

    cout << "Training ..." << endl;
    cout << "===========================================" << endl;

    auto begin_train_time = std::chrono::high_resolution_clock::now();
    std::vector<double> epoch_mses;

    // Run EPOCH number of cycles for optimizing the solution
    for (int epoch = 1; epoch <= trainingEpochs; epoch++) {
        // Train neural network. If this is the first iteration, weights are random,
        // using current values as starting point otherwise.
        model.Train(trainX, trainY, optimizer);
        //model.Parameters().print(std::cout);

        optimizer.ResetPolicy() = false;

        arma::cube predOut;

        // Getting predictions on test data points.
        model.Predict(testX, predOut);

        // Calculating MSE on test data points.
        double testMSE = MSECalc(predOut, testY);
        epoch_mses.push_back(testMSE);

        // Print stats during training
        if (epoch % 10 == 0 || epoch == 1)
        cout << "|=== [Epoch: " << epoch << "\t|\tAccuracy: " << setprecision(2) << fixed << (100 - testMSE)
             << " %] ===|" << endl;
    }

    cout << "===========================================" << endl;
    // End of measuring training time
    auto end_train_time = std::chrono::high_resolution_clock::now();
    auto train_time = end_train_time - begin_train_time;
    cout << "Training ... OK." << endl;

    cout << "Average accuracy during training: " << fixed << setprecision(2)
         << (100 - TakeVectorAVG(epoch_mses)) << " %" << endl;

    if (train_time.count() / 1e+9 < 60)
        cout << "Training time: " << setprecision(2) << fixed << train_time.count() / 1e+9 << " second(s)."
             << endl;
    else
        cout << "Training time: " << setprecision(1) << train_time.count() / 6e+10 << " minute(s)." << endl;

    cout << "Saving Model ...";
    data::Save(modelFile, "eyeState", model);
    cout << " OK." << endl;

}

void RNNPredictor::MakePrediction() {

    // Load RNN model and use it for prediction.
    RNN<MeanSquaredError<>, HeInitialization> modelP(rho);
    cout << "Loading model ...";
    data::Load(modelFile, "eyeState", modelP);
    cout << " OK." << endl;
    arma::cube predOutP;

    cout << "Predicting ...";
    // Get predictions on test data points.
    modelP.Predict(testX, predOutP);
    cout << " OK." << endl;
    // Calculate MSE on prediction.
    double testMSEPred = MSECalc(predOutP, testY);
    cout << "Prediction Accuracy: " << setprecision(2) << fixed << (100 - testMSEPred) << " %" << endl;

}
