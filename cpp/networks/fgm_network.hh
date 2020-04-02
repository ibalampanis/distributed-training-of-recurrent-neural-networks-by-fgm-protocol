#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FGM_NETWORK_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FGM_NETWORK_HH

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"

namespace fgm_network {

    using namespace dds;
    using namespace gm_protocol;
    using namespace rnn_learner;
    using std::map;
    using std::cout;
    using std::endl;


    struct Coordinator;
    struct CoordinatorProxy;
    struct LearningNode;
    struct LearningNodeProxy;

    /**
     * This is the FGM Network implementation.
     */
    struct FgmNet : GmLearningNetwork<FgmNet, Coordinator, LearningNode> {
        typedef GmLearningNetwork<network_t, coordinator_t, node_t> gm_learning_network_t;

        FgmNet(const set<source_id> &_hids, const string &_name, Query *_Q);
    };

    /**
    * This is the hub/coordinator implementation for the Functional Geometric Method protocol.
    */
    struct Coordinator : process {
        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef FgmNet network_t;

        proxy_map<node_proxy_t, node_t> proxy;

        /** Protocol Stuff */
        RnnLearner *globalLearner;
        Query *Q;                 // continuous query
        QueryState *query;                      // current query state
        SafezoneFunction *safezone;          // the safe zone wrapper
        size_t k;                                  // number of sites

        /** Nodes indexing */
        map<node_t *, size_t> nodeIndex;
        map<node_t *, size_t> nodeBoolDrift;
        vector<node_t *> nodePtr;

        float phi;                       // The phi value of the functional geometric protocol.
        float quantum;                   // The quantum of the functional geometric protocol.
        size_t counter;                  // A counter used by the functional geometric protocol.
        float barrier;                   // The smallest number the zeta function can reach.
        size_t cnt;                      // Helping counter.
        bool rebalanced;                 // Flag to check if the current round has be rebalanced.
        size_t roundRebs;                // The number of rebalances ocuured in this round.

        arma::mat params;        // A placeholder for the parameters send by the nodes.
        vector<arma::mat> beta;          // The beta vector used by the protocol for the rebalancing process.

        /** Statistics */
        size_t numRounds;                   // Total number of rounds
        size_t numSubrounds;                // Total number of subrounds
        size_t szSent;                      // Total safe zones sent
        size_t totalUpdates;                // Number of stream updates received
        size_t numRebalances;               // Number of rebalances

        /** Constructor and Destructor */
        Coordinator(network_t *nw, Query *_Q);

        ~Coordinator();

        network_t *Net();

        const ProtocolConfig &Cfg() const;

        /** Initialize the Learner and its' variables */
        void InitializeGlobalLearner();

        /** Method used by the hub to establish the connections with the nodes of the star network */
        void SetupConnections();

        /** Initialize a new round */
        void StartRound();

        /** Getting the model of a node */
        void FetchUpdates(node_t *n);

        /** Remote call on host violation */
        oneway SendIncrement(Increment inc);

        oneway Drift(sender<node_t> ctx, size_t cols);

        void FinishRound();

        void FinishRounds();

        /** Rebalancing method */
        void Rebalance();

        /** Printing and saving the accuracy */
        void Progress();

        /** Getting the accuracy of the global learner */
        double Accuracy();

        /** Get the communication statistics of experiment */
        vector<size_t> Statistics() const;

    };

    struct CoordinatorProxy : remote_proxy<Coordinator> {
        using coordinator_t = Coordinator;

        REMOTE_METHOD(coordinator_t, SendIncrement);
        REMOTE_METHOD(coordinator_t, Drift);

        CoordinatorProxy(process *c) : remote_proxy<coordinator_t>(c) {}
    };

    /**
    * This is the site/learning node implementation for the Functional Geometric Method protocol.
    */
    struct LearningNode : local_site {

        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef FgmNet network_t;
        typedef CoordinatorProxy coord_proxy_t;
        typedef Query continuous_query_t;

        Query *Q;                 // The query management object.
        Safezone szone;                      // The safezone object.
        RnnLearner *_learner;            // The learning algorithm.
        arma::mat deltaVector;
        arma::mat eDelta;

        size_t numSites;                     // Number of sites.
        coord_proxy_t coord;                 // The proxy of the coordinator/hub.

        size_t counter;                      // The counter used by the FGM protocol.
        float quantum;                       // The quantum provided by the hub.
        float zeta;                          // The value of the safezone function.
        bool rebalanced;                     // Flag to check if the current round has be rebalanced.

        LearningNode(network_t *net, source_id hid, continuous_query_t *_Q);

        const ProtocolConfig &Cfg() const;

        void InitializeLearner();

        void SetupConnections();

        void UpdateStream(arma::mat &batch, arma::mat &labels);

        void UpdateDrift(vector<arma::mat *> &params);

        /**
         * Remote Methods
         */

        // called at the start of a round
        oneway Reset(const Safezone &newsz, const FloatValue qntm);

        // Refreshing the quantum for a new subround
        oneway TakeQuantum(const FloatValue qntm);

        // Refreshing the quantum and reverting the drift vector back to E
        oneway Rebalance(const FloatValue qntm);

        // Transfer data to the coordinator
        ModelState GetDrift();

        // Transfer the value of z(Xi) to the coordinator
        FloatValue GetZetaValue();

        // Loading the parameters of the hub to the local model.
        oneway SetGlobalParameters(const ModelState &params);

    };

    struct LearningNodeProxy : remote_proxy<LearningNode> {
        typedef LearningNode node_t;
        REMOTE_METHOD(node_t, Reset);
        REMOTE_METHOD(node_t, TakeQuantum);
        REMOTE_METHOD(node_t, Rebalance);
        REMOTE_METHOD(node_t, GetDrift);
        REMOTE_METHOD(node_t, GetZetaValue);
        REMOTE_METHOD(node_t, SetGlobalParameters);

        explicit LearningNodeProxy(process *p) : remote_proxy<node_t>(p) {}
    };


} // end namespace fgm_network

namespace dds {
    template<>
    inline size_t byte_size<fgm_network::LearningNode *>(
            fgm_network::LearningNode *const &) { return 4; }
}
#endif
