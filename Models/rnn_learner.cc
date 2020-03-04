#include <mlpack/core.hpp>
#include <jsoncpp/json/json.h>
#include <iostream>
#include "rnn_learner.hh"


using namespace rnn_learner;
using namespace arma;

RNNLearner::RNNLearner(string cfg) {
    try {
        std::ifstream cfgfile(cfg);
        cfgfile >> root;
        string temp = root["hyperparameters"].get("trainingEpochs", 0).asString();
        trainingEpochs = std::stoi(temp);
        lstmCells = root["hyperparameters"].get("lstmCells", 0).asInt();
        rho = root["hyperparameters"].get("rho", 0).asInt();
        stepSize = root["hyperparameters"].get("stepSize", 0).asDouble();
        batchSize = root["hyperparameters"].get("batchSize", 0).asInt();
        iterationsPerEpoch = root["hyperparameters"].get("iterationsPerEpoch", 0).asInt();
        tolerance = root["hyperparameters"].get("tolerance", 0).asDouble();
        bShuffle = root["hyperparameters"].get("bShuffle", 0).asBool();
        epsilon = root["hyperparameters"].get("epsilon", 0).asDouble();
        beta1 = root["hyperparameters"].get("beta1", 0).asDouble();
        beta2 = root["hyperparameters"].get("beta2", 0).asDouble();
        trainTestRatio = root["hyperparameters"].get("trainTestRatio", 0).asDouble();
        datasetPath = root["data"].get("path", "").asString();
        maxRho = rho;
        numberOfUpdates = 0;
    } catch (...) {
        throw;
    }
}

void RNNLearner::CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, const size_t rho) {
    for (size_t i = 0; i < dataset.n_cols - rho; i++) {
        X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(), arma::span(i, i + rho - 1));
        y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(
                arma::span(dataset.n_rows - 1, dataset.n_rows - 1), arma::span(i + 1, i + rho));
    }
}

double RNNLearner::TakeVectorAVG(const std::vector<double> &vec) {
    double sum = 0;
    for (double i : vec)
        sum += i;

    double avg = sum / vec.size();
    return avg;
}

double RNNLearner::CalcMSE(arma::cube &pred, arma::cube &Y) {
    double err_sum = 0.0;
    arma::cube diff = pred - Y;
    for (size_t i = 0; i < diff.n_slices; i++) {
        arma::mat temp = diff.slice(i);
        err_sum += accu(temp % temp);
    }
    return (err_sum / (diff.n_elem + 1e-50));
}

double RNNLearner::GetModelAccuracy() const { return modelAccuracy; }

int RNNLearner::GetNumberOfUpdates() const {
    return numberOfUpdates;
}

const Mat<double> &RNNLearner::GetModelParameters() const {
    return modelParameters;
}

void RNNLearner::SetModelParameters(const Mat<double> &modelParameters) {
    RNNLearner::modelParameters = modelParameters;
}

void RNNLearner::CentralizedDataPreparation() {

    arma::mat dataset;
    // In Armadillo rows represent features, columns represent data points.
    cout << "Reading dataset ...";
    data::Load(datasetPath, dataset, true);
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

void RNNLearner::TrainModel() {

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

        optimizer.ResetPolicy() = false;

        arma::cube predOut;

        // Getting predictions on test data points.
        model.Predict(testX, predOut);

        // Calculating MSE and accuracy on test data points.
        double testMSE = CalcMSE(predOut, testY);
        modelAccuracy = 100 - testMSE;
//        modelParameters = static_cast<mat>(model.Parameters());

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

void RNNLearner::MakePrediction() {

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
    double testMSEPred = CalcMSE(predOutP, testY);
    cout << "Prediction Accuracy: " << setprecision(2) << fixed << (100 - testMSEPred) << " %" << endl;

}

