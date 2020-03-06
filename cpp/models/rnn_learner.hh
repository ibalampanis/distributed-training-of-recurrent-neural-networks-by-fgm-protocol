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

namespace rnn_learner {


    using namespace std;
    using namespace mlpack;
    using namespace mlpack::ann;
    using namespace ens;
    using namespace arma;

    class RNNLearner {

    protected:
        arma::cube trainX, trainY;              // Trainset data points and labels
        arma::cube testX, testY;                // Testset data points and labels
        size_t inputSize;                       // Number of neurons at the input layer
        size_t outputSize;                      // Number of neurons at the output layer
        string datasetPath;                     // Path for finding dataset file
        string saveModelPath;                   // Path for saving object of rnn model
        double trainTestRatio;                  // Testing data is taken from the dataset in this ratio
        int trainingEpochs;                     // Number of optimization epochs
        int lstmCells;                          // Number of hidden layers
        int rho;                                // Number of time steps to look backward for in the RNN
        int maxRho = rho;                       // Max Rho for LSTM
        double stepSize;                        // Step size of an optimizer
        int batchSize;                          // Number of data points in each iteration of SGD
        int iterationsPerEpoch;                 // Number of iterations per cycle
        double tolerance;                       // Optimizer tolerance
        bool bShuffle;                          // Let optimizer shuffle batches
        double epsilon;                         // Optimizer epsilon
        double beta1;                           // Optimizer beta1
        double beta2;                           // Optimizer beta2
        double modelAccuracy;                   // Current accuracy of model
        vector<arma::mat *> modelParameters;    // Current model parameters
        size_t numberOfUpdates;                 // A counter for parameters updates
        Json::Value root;                       // JSON file to read the hyperparameters

    public:

        /**
         * Constructor and Destructor
         * **/
        explicit RNNLearner(const string &cfg);

        ~RNNLearner();

        static void CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, size_t rho);

        static double CalcMSE(arma::cube &pred, arma::cube &Y);

        double GetModelAccuracy() const;

        int GetNumberOfUpdates() const;

        vector<arma::mat *> &GetModelParameters();

        void SetModelParameters(vector<arma::mat *> &modelParameters);

        void CentralizedDataPreparation();

        void TrainModel();

        void MakePrediction();


    };
}

#endif //DISTRIBUTEDRNNS_RNNPREDICTOR_HH