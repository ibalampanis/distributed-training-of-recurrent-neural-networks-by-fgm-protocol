#ifndef DISTRIBUTEDRNNS_RNNPREDICTOR_HH
#define DISTRIBUTEDRNNS_RNNPREDICTOR_HH

#include <iostream>
#include <chrono>
#include <iomanip>
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <ensmallen.hpp>

namespace rnn_predictor {


    using namespace std;
    using namespace mlpack;
    using namespace mlpack::ann;
    using namespace ens;

    class RNNPredictor {

    private:
        double trainTestRatio = 0.3;        // Testing data is taken from the dataset in this ratio.
        int trainingEpochs = 20;            // Number of optimization epochs.
        int lstmCells = 30;                 // Number of hidden layers.
        int rho = 25;                       // Number of time steps to look backward for in the RNN.
        const int maxRho = rho;             // Max Rho for LSTM.
        double stepSize = 4.5e-5;           // Step size of an optimizer.
        int batchSize = 128;                // Number of data points in each iteration of SGD
        int iterationsPerEpoch = 10000;     // Number of iterations per cycle.
        double tolerance = 1e-8;            // Optimizer tolerance
        bool bShuffle = true;               // Let optimizer shuffle batches
        double epsilon = 2e-8;              // Optimizer epsilon
        double beta1 = 0.9;                 // Optimizer beta1
        double beta2 = 0.999;               // Optimizer beta2
        double modelAccuracy;               // Current accuracy of model

    public:
        RNNPredictor(int trainingEpochs, int lstmCells, int rho, double stepSize, int batchSize,
                     int iterationsPerEpoch);

        const string dataFile = "../data/EEG_Eye_State.csv";
        const string modelFile = "../saved_models/eyeState.bin";
        size_t inputSize = 15, outputSize = 1;
        arma::cube trainX, trainY, testX, testY;


        static void createTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, const size_t rho);

        static double takeVectorAVG(const std::vector<double> &vec);

        static double calcMSE(arma::cube &pred, arma::cube &Y);

        double getModelAccuracy() const;

        void DataPreparation();

        void TrainModel();

        void MakePrediction();
    };
}

#endif //DISTRIBUTEDRNNS_RNNPREDICTOR_HH