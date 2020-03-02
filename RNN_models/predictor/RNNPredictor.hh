#ifndef DISTRIBUTEDRNNS_RNNPREDICTOR_HH
#define DISTRIBUTEDRNNS_RNNPREDICTOR_HH

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <jsoncpp/json/json.h>
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

    protected:
        double trainTestRatio;                  // Testing data is taken from the dataset in this ratio.
        int trainingEpochs;                     // Number of optimization epochs.
        int lstmCells;                          // Number of hidden layers.
        int rho;                                // Number of time steps to look backward for in the RNN.
        int maxRho = rho;                       // Max Rho for LSTM.
        double stepSize;                        // Step size of an optimizer.
        int batchSize;                          // Number of data points in each iteration of SGD
        int iterationsPerEpoch;                 // Number of iterations per cycle.
        double tolerance;                       // Optimizer tolerance
        bool bShuffle;                          // Let optimizer shuffle batches
        double epsilon;                         // Optimizer epsilon
        double beta1;                           // Optimizer beta1
        double beta2;                           // Optimizer beta2
        double modelAccuracy;                   // Current accuracy of model
        arma::Mat<double> modelParameters;      // Current model parameters
        size_t numberOfUpdates;                 // A counter for parameters updates
        Json::Value root;                       // JSON file to read the hyperparameters

    public:
        /** Constructor **/
        RNNPredictor(string cfg, string net_name);

        const string dataFile = "../data/EEG_Eye_State.csv";
        const string modelFile = "../saved_models/eyeState.bin";
        size_t inputSize = 15, outputSize = 1;
        arma::cube trainX, trainY, testX, testY;


        static void createTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, size_t rho);

        static double takeVectorAVG(const std::vector<double> &vec);

        static double calcMSE(arma::cube &pred, arma::cube &Y);

        double getModelAccuracy() const;

        int getNumberOfUpdates() const;

        const arma::Mat<double> &getModelParameters() const;

        void setModelParameters(const arma::Mat<double> &modelParameters);

        void CentralizedDataPreparation();

        void TrainModel();

        void MakePrediction();


    };
}

#endif //DISTRIBUTEDRNNS_RNNPREDICTOR_HH