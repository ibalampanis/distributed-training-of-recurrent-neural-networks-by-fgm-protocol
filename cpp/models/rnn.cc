#include <mlpack/core.hpp>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <utility>
#include "rnn.hh"

using namespace rnn;
using namespace arma;
using namespace std;

RnnLearner::RnnLearner(const string &cfg, const RNN<MeanSquaredError<>, HeInitialization> &model) : model(model) {

    // Take values from JSON file and initialize parameters
    try {
        ifstream cfgfile(cfg);
        cfgfile >> root;
        trainingEpochs = root["hyperparameters"].get("training_epochs", -1).asInt();
        lstmCells = root["hyperparameters"].get("lstm_cells", -1).asInt();
        lstmLayers = root["hyperparameters"].get("lstm_layers", -1).asInt();
        rho = root["hyperparameters"].get("rho", -1).asInt();
        stepSize = root["hyperparameters"].get("step_size", -1).asDouble();
        batchSize = root["hyperparameters"].get("batch_size", -1).asInt();
        maxOptIterations = root["hyperparameters"].get("max_opt_iterations", -1).asInt();
        tolerance = root["hyperparameters"].get("tolerance", -1).asDouble();
        bShuffle = root["hyperparameters"].get("shuffle", 0).asBool();
        epsilon = root["hyperparameters"].get("epsilon", -1).asDouble();
        beta1 = root["hyperparameters"].get("beta1", -1).asDouble();
        beta2 = root["hyperparameters"].get("beta2", -1).asDouble();
        trainTestRatio = root["hyperparameters"].get("train_test_ratio", -1).asDouble();
        vocabSize = root["data"].get("vocab_size", -1).asInt();
        embedSize = root["data"].get("embed_size", -1).asInt();
        inputSize = root["data"].get("input_size", -1).asInt();
        outputSize = root["data"].get("output_size", -1).asInt();
        datasetType = root["data"].get("type", "").asString();
        datasetPath = root["data"].get("path", "").asString();
        featsPath = root["data"].get("feat_path", "").asString();
        labelsPath = root["data"].get("labels_path", "").asString();
    } catch (...) {}
    numberOfUpdates = 0;
    usedTimes = 0;
    maxRho = rho;
    modelAccuracy = 0.0;
}

RnnLearner::~RnnLearner() = default;

void RnnLearner::CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, size_t rho) {
    for (size_t i = 0; i < dataset.n_cols - rho; i++) {
        X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(), arma::span(i, i + rho - 1));
        y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(
                arma::span(dataset.n_rows - 1, dataset.n_rows - 1), arma::span(i + 1, i + rho));
    }
}

void RnnLearner::CreateTimeSeriesData(arma::mat feats, arma::mat labels, arma::cube &X, arma::cube &y, size_t rho) {
    for (size_t i = 0; i < feats.n_cols - rho; i++) {
        X.subcube(arma::span(), arma::span(i), arma::span()) = feats.submat(arma::span(), arma::span(i, i + rho - 1));
        y.subcube(arma::span(), arma::span(i), arma::span()) = labels.submat(
                arma::span(labels.n_rows - 1, labels.n_rows - 1), arma::span(i + 1, i + rho));
    }
}

double RnnLearner::CalculateMSPE(arma::cube &pred, arma::cube &Y) {
    return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

size_t RnnLearner::NumberOfUpdates() const { return numberOfUpdates; }

size_t RnnLearner::UsedTimes() const { return usedTimes; }

arma::mat RnnLearner::ModelParameters() const { return model.Parameters(); }

void RnnLearner::UpdateModel(arma::mat params) { model.Parameters() = move(params); }

double RnnLearner::ModelAccuracy() const { return modelAccuracy; }

void RnnLearner::CentralizedDataPreparation() {

    arma::cube X, y;

    if (datasetType != "sep") {
        arma::mat dataset;
        // In Armadillo rows represent features, columns represent data points.
//        cout << "Reading dataset ...";
        data::Load(datasetPath, dataset, true);
//        cout << " OK." << endl;

        // Scale all data into the range (0, 1) for increased numerical stability.
        data::MinMaxScaler scale;
        scale.Fit(dataset);
        scale.Transform(dataset, dataset);

        // We need to represent the input data for RNN in an arma::cube (3D matrix).
        // The 3rd dimension is rho, the number of past data records the RNN uses for learning.
        X.set_size(inputSize, dataset.n_cols - rho + 1, rho);
        y.set_size(outputSize, dataset.n_cols - rho + 1, rho);

        CreateTimeSeriesData(dataset, X, y, rho);
    } else {
        arma::mat features, labels;

        // In Armadillo rows represent features, columns represent data points.
//        cout << "Reading dataset ...";
        data::Load(featsPath, features, true);
        data::Load(labelsPath, labels, true);
//        cout << " OK." << endl;


        // We need to represent the input data for RNN in an arma::cube (3D matrix).
        // The 3rd dimension is rho, the number of past data records the RNN uses for learning.
        X.set_size(inputSize, features.n_cols - rho + 1, rho);
        y.set_size(outputSize, labels.n_cols - rho + 1, rho);

        CreateTimeSeriesData(features, labels, X, y, rho);

        features.clear();
        labels.clear();
    }

    // Split the data into training and testing sets.
    size_t trainingSize = (1 - trainTestRatio) * X.n_cols;
    trainX = X.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
    trainY = y.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
    testX = X.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());
    testY = y.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());
}

void RnnLearner::BuildModel() {
    // Model definition
    model = RNN<MeanSquaredError<>, HeInitialization>(rho);

    // Model building
    model.Add<IdentityLayer<> >();
//    if (datasetType == "nlp")
//        model.Add<Embedding<> >(vocabSize, embedSize);
    for (int k = 0; k < lstmLayers; k++) {
        model.Add<LSTM<> >(inputSize, lstmCells, maxRho);
        model.Add<Dropout<> >(0.5);
        model.Add<ReLULayer<> >();
    }
    model.Add<Linear<> >(lstmCells, outputSize);

    // Define and set parameters for the Stochastic Gradient Descent (SGD) optimizer. 
    optimizer = SGD<AdamUpdate>(stepSize, batchSize, maxOptIterations, tolerance, bShuffle,
                                AdamUpdate(epsilon, beta1, beta2));

}

void RnnLearner::TrainModel() {


    cout << "Training ..." << endl;

    auto begin_train_time = chrono::high_resolution_clock::now();

    cout << "lstm size: " << lstmCells << " layers: " << lstmLayers << " learn rate: " << stepSize << endl;

    // Run EPOCH number of cycles for optimizing the solution
    for (size_t epoch = 1; epoch <= trainingEpochs; epoch++) {
        // Train neural network. If this is the first iteration, weights are random,
        // using current values as starting point otherwise.
        model.Train(trainX, trainY, optimizer);


        optimizer.ResetPolicy() = false;

        arma::cube predOut;

        // Getting predictions on test data points.
        model.Predict(testX, predOut);

        // Calculating MSE and accuracy on test data points.
        double testMSE = CalculateMSPE(predOut, testY);
        modelAccuracy = 100 - testMSE;

        // Print stats during training
        if ((trainingEpochs > 10 && epoch % 10 == 0) || epoch == 1)
            cout << "Epoch: " << epoch << "/" << trainingEpochs << "\t|\tAccuracy: " << setprecision(2) << fixed
                 << (100 - testMSE) << " %" << endl;
        else
            cout << "Epoch " << epoch << " from " << trainingEpochs << "\t|\tAccuracy: " << setprecision(2) << fixed
                 << (100 - testMSE) << " %" << endl;
    }

    // End of measuring training time
    auto end_train_time = chrono::high_resolution_clock::now();
    auto train_time = end_train_time - begin_train_time;
    cout << "Training ... OK." << endl;

    if (train_time.count() / 1e+9 < 60)
        cout << "Training time: " << setprecision(2) << fixed << train_time.count() / 1e+9 << " second(s)."
             << endl;
    else
        cout << "Training time: " << setprecision(1) << train_time.count() / 6e+10 << " minute(s)." << endl;

}

void RnnLearner::TrainModelByBatch(arma::cube &x, arma::cube &y) {
    model.Train(x, y, optimizer);
    numberOfUpdates += x.n_cols;
    usedTimes++;
    optimizer.ResetPolicy() = false;
}

void RnnLearner::MakePrediction() {

    arma::cube predOut;
//    cout << "Predicting ...";
    // Get predictions on test data points.
    model.Predict(testX, predOut);
//    cout << " OK." << endl;
    // Calculate MSE on prediction.
    double testMSEPred = CalculateMSPE(predOut, testY);
    cout << "Prediction Accuracy: " << setprecision(2) << fixed << (100 - testMSEPred) << " %" << endl;
}

double RnnLearner::MakePrediction(arma::cube &tX, arma::cube &tY) {
    arma::cube predOut;
    model.Predict(tX, predOut);
    return (100 - CalculateMSPE(predOut, tY));
}
