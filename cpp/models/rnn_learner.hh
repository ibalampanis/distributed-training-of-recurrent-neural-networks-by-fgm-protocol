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

    class RnnLearner {

    protected:

        //  Dataset parameters 
        arma::cube trainX, trainY;                          // Trainset data points and labels
        arma::cube testX, testY;                            // Testset data points and labels
        size_t inputSize;                                   // Number of neurons at the input layer
        size_t outputSize;                                  // Number of neurons at the output layer
        string datasetPath;                                 // Path for finding dataset file
        double trainTestRatio;                              // Testing data is taken from the dataset in this ratio

        RNN<MeanSquaredError<>, HeInitialization> model;    // RNN model
        SGD<AdamUpdate> optimizer;                          // SGD optimizer

        //  Model and Optimizer parameters 
        size_t trainingEpochs;                              // Number of optimization epochs
        size_t lstmCells;                                   // Number of hidden layers
        size_t rho;                                         // Number of time steps to look backward for in the RNN
        size_t maxRho = rho;                                // Max Rho for LSTM
        double stepSize;                                    // Step size of an optimizer
        size_t batchSize;                                   // Number of data points in each iteration of SGD
        size_t iterationsPerEpoch;                          // Number of iterations per cycle
        double tolerance;                                   // Optimizer tolerance
        bool bShuffle;                                      // Let optimizer shuffle batches
        double epsilon;                                     // Optimizer epsilon
        double beta1;                                       // Optimizer beta1
        double beta2;                                       // Optimizer beta2

        size_t numberOfUpdates;                             // Number of model updates
        double modelAccuracy;                               // Current accuracy of model
        Json::Value root;                                   // JSON file to read the hyperparameters

    public:

        //  Constructor and Destructor 
        explicit RnnLearner(const string &cfg, const RNN<MeanSquaredError<>, HeInitialization> &model);

        ~RnnLearner();

        static void CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y, size_t rho);

        static double CalcMSE(arma::cube &pred, arma::cube &Y);

        size_t NumberOfUpdates() const;

        arma::mat ModelParameters() const;

        void UpdateModel(arma::mat params);

        double ModelAccuracy() const;

        void CentralizedDataPreparation();

        void BuildModel();

        void TrainModel();

        void TrainModelByBatch(arma::cube &x, arma::cube &y);

        void MakePrediction();

        double MakePrediction(arma::cube &tX, arma::cube &tY);

    };

} // end namespace rnn_learner

#endif //DISTRIBUTEDRNNS_RNNPREDICTOR_HH