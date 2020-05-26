#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FGM_NETWORK_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FGM_NETWORK_HH

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include "protocols.hh"
#include "cpp/models/rnn.hh"


namespace algorithms {

    namespace fgm {

        using namespace dds;
        using namespace protocols;
        using namespace rnn;

        struct Coordinator;
        struct CoordinatorProxy;
        struct LearningNode;
        struct LearningNodeProxy;

        // This is the FGM Network implementation.
        struct FgmNet : LearningNetwork<FgmNet, Coordinator, LearningNode> {
            typedef LearningNetwork<network_t, coordinator_t, node_t> fgmLearningNetwork;

            FgmNet(const set<source_id> &_hids, const string &_name, Query *Q);
        };

        // This is the hub/coordinator implementation for the Functional Geometric Method protocol.
        struct Coordinator : process {
            typedef Coordinator coordinator_t;
            typedef LearningNode node_t;
            typedef LearningNodeProxy node_proxy_t;
            typedef FgmNet network_t;

            proxy_map<node_proxy_t, node_t> proxy;

            // Protocol Stuff
            RnnLearner *globalLearner;          // ML model
            arma::cube trainX, trainY;          // Trainset data points and labels for warmup
            arma::cube testX, testY;            // Testset data points and labels
            size_t trainPoints;
            Query *Q;                           // query
            QueryState *queryState;             // current query state
            SafeFunction *safeFunction;         // the safe zone wrapper
            size_t k;                           // number of sites

            // Nodes indexing
            map<node_t *, size_t> nodeIndex;
            vector<node_t *> nodePtr;

            double psi;                         // The psi value of the functional geometric protocol.
            double theta;                       // The quantum of the functional geometric protocol.
            size_t counter;                     // A counter used by the functional geometric protocol.
            double barrier;                     // The smallest number the zeta function can reach.
            arma::mat params;                   // A placeholder for the parameters send by the nodes.

            // Statistics
            size_t nRounds;                     // Total number of rounds
            size_t nSubrounds;                  // Total number of subrounds
            size_t nSzSent;                     // Total safe zones sent
            size_t nUpdates;                    // Number of stream updates received

            // Constructor and Destructor
            Coordinator(network_t *nw, Query *Q);

            ~Coordinator() override;

            network_t *Net();

            const ProtocolConfig &Cfg() const;

            // Initialize the Learner and its' variables
            void InitializeGlobalLearner();

            void WarmupGlobalLearner();

            // Method used by the hub to establish the connections with the nodes of the star network
            void SetupConnections();

            // Initialize a new round
            void StartRound();

            // Getting the model of a node
            void FetchUpdates(node_t *n);

            // Remote call on host violation
            oneway ReceiveIncrement(IntValue inc);

            void FinishRound();

            void ShowOverallStats();

            // Printing and saving the accuracy
            void ShowProgress();

            // Getting the accuracy of the global learner
            double Accuracy();
        };

        struct CoordinatorProxy : remote_proxy<Coordinator> {
            using coordinator_t = Coordinator;

            REMOTE_METHOD(coordinator_t, ReceiveIncrement);

            explicit CoordinatorProxy(process *c) : remote_proxy<coordinator_t>(c) {}
        };

        // This is the site/learning node implementation for the Functional Geometric Method protocol.
        struct LearningNode : local_site {

            typedef Coordinator coordinator_t;
            typedef LearningNode node_t;
            typedef LearningNodeProxy node_proxy_t;
            typedef FgmNet network_t;
            typedef CoordinatorProxy coord_proxy_t;
            typedef Query continuous_query_t;

            Query *Q;                           // The query management object.
            Safezone szone;                     // The safezone object.
            RnnLearner *learner;                // The learning algorithm.
            coord_proxy_t coord;                // The proxy of the coordinator/hub.
            arma::mat drift;
            arma::mat currentEstimate;          // The global estimate. It has a fixed value during a round.
            size_t localCounter;                // The counter used by the FGM protocol.
            float theta;                        // The quantum provided by the hub.
            float zeta;                         // The value of the safezone function.
            float phi;

            size_t datapointsPassed;

            LearningNode(network_t *net, source_id hid, continuous_query_t *Q);

            const ProtocolConfig &Cfg() const;

            void InitializeLearner();

            void UpdateState(arma::cube &x, arma::cube &y);

            // Called at the start of each round
            oneway ResetForNewRound(const Safezone &newsz, DoubleValue qntm);

            // Refreshing the quantum for a new subround
            oneway ReceiveQuantum(DoubleValue qntm);

            // Transfer data to the coordinator
            ModelState SendDrift();

            // Transfer the value of z(Xi) to the coordinator
            DoubleValue SendZeta();

            // Loading the parameters of the hub to the local model.
            oneway ReceiveGlobalModel(const ModelState &params);
        };

        struct LearningNodeProxy : remote_proxy<LearningNode> {
            typedef LearningNode node_t;

            REMOTE_METHOD(node_t, ResetForNewRound);
            REMOTE_METHOD(node_t, ReceiveQuantum);
            REMOTE_METHOD(node_t, SendDrift);
            REMOTE_METHOD(node_t, SendZeta);
            REMOTE_METHOD(node_t, ReceiveGlobalModel);

            explicit LearningNodeProxy(process *p) : remote_proxy<node_t>(p) {}
        };

    } // end namespace fgm
} // end namespace algorithms


namespace dds {
    template<>
    inline size_t byte_size<algorithms::fgm::LearningNode *>(
            algorithms::fgm::LearningNode *const &) { return 4; }
}
#endif
