#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_HH

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include "protocols.hh"
#include "cpp/models/rnn_learner.hh"

namespace gm_network {

    using namespace dds;
    using namespace rnn_learner;
    using namespace gm_protocol;

    struct Coordinator;
    struct CoordinatorProxy;
    struct LearningNode;
    struct LearningNodeProxy;

    // This is the GM Network implementation for the classic Geometric Method protocol. 
    struct GmNet : gm_protocol::GmLearningNetwork<GmNet, Coordinator, LearningNode> {

        typedef gm_protocol::GmLearningNetwork<network_t, coordinator_t, node_t> gm_learning_network_t;

        GmNet(const set<source_id> &_hids, const string &_name, Query *_Q);
    };

    // This is the hub/coordinator implementation for the classic Geometric Method protocol. 
    struct Coordinator : process {
        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef GmNet network_t;

        proxy_map<node_proxy_t, node_t> proxy;

        // Protocol Stuff 
        RnnLearner *globalLearner;          // ML model
        arma::cube testX, testY;            // Testset data points and labels
        Query *Q;                           // query
        QueryState *query;                  // current query state
        SafezoneFunction *safezone;         // the safe zone wrapper
        size_t k;                           // number of sites
        map<node_t *, size_t> nodeIndex;    // index the nodes
        vector<node_t *> nodePtr;
        set<node_t *> B;                    // initialized by local_violation(), updated by rebalancing algorithm
        set<node_t *> Bcompl;               // Complement of B, updated by rebalancing algo
        arma::mat Mean;                     // Used to compute the mean model
        size_t numViolations;               // Number of violations in the same round (for rebalancing)
        int cnt;                            // Helping counter.

        // Statistics 
        size_t numRounds;                   // Total number of rounds
        size_t numSubrounds;                // Total number of subrounds
        size_t szSent;                      // Total safe zones sent
        size_t totalUpdates;                // Number of stream updates received

        // Constructor and Destructor 
        Coordinator(network_t *nw, Query *_Q);

        ~Coordinator() override;

        network_t *Net();

        const ProtocolConfig &Cfg() const;

        // Initialize learner and  variables 
        void InitializeGlobalLearner();

        void SetupConnections();

        // Initialize a new round 
        void StartRound();

        // Rebalance algorithm by Kamp
        void Rebalance(node_t *n);

        // Get a model of a node
        void FetchUpdates(node_t *n);

        // Remote call on host violation
        oneway LocalViolation(sender<node_t> ctx);

        // Printing and saving the accuracy
        void ShowProgress();

        // Get the communication statistics of experiment
        vector<size_t> UpdateStats() const;

        void FinishRound();

        // Getting the accuracy of the global learner
        double Accuracy();

        void ShowOverallStats();
    };

    struct CoordinatorProxy : remote_proxy<Coordinator> {
        using coordinator_t = Coordinator;

        REMOTE_METHOD(coordinator_t, LocalViolation);

        explicit CoordinatorProxy(process *c) : remote_proxy<coordinator_t>(c) {}
    };

    // This is the site/learning node implementation for the classic Geometric Method protocol. 
    struct LearningNode : local_site {

        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef GmNet network_t;
        typedef CoordinatorProxy coord_proxy_t;
        typedef Query query_t;

        query_t *Q;                         // The query management object
        Safezone szone;                     // The safezone object
        RnnLearner *learner;                // The learning algorithm
        arma::cube trainX, trainY;          // Trainset data points and labels
        arma::cube testX, testY;            // Testset data points and labels
        arma::mat drift;                    // The drift of the node
        coord_proxy_t coord;                // The proxy of the coordinator/hub
        size_t datapointsPassed;

        // Constructor 
        LearningNode(network_t *net, source_id hid, query_t *_Q);

        const ProtocolConfig &Cfg() const;

        void InitializeLearner();

        void UpdateState(arma::cube &x, arma::cube &y);

        // Call at the start of a round 
        oneway Reset(const Safezone &newsz);

        // Transfer data to the coordinator 
        ModelState SendDrift();

        // Set the drift vector (for rebalancing) 
        void ReceiveDrift(const ModelState &mdl);

        oneway ReceiveGlobalParameters(const ModelState &params);

    };

    struct LearningNodeProxy : remote_proxy<gm_network::LearningNode> {
        typedef gm_network::LearningNode node_t;
        REMOTE_METHOD(node_t, Reset);
        REMOTE_METHOD(node_t, SendDrift);
        REMOTE_METHOD(node_t, ReceiveDrift);
        REMOTE_METHOD(node_t, ReceiveGlobalParameters);

        explicit LearningNodeProxy(process *p) : remote_proxy<node_t>(p) {}
    };


} // end namespace gm_network

namespace dds {
    template<>
    inline size_t byte_size<gm_network::LearningNode *>(
            gm_network::LearningNode *const &) { return 4; }
}

#endif