#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_NETWORK_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_NETWORK_HH

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"

namespace gm_protocol {

    using namespace dds;
    using namespace rnn_learner;
    using std::map;
    using std::cout;
    using std::endl;


    struct NodeProxy;
    struct Coordinator;
    struct LearningNode;
    struct LearningNodeProxy;

    struct GM_Net : GmLearningNetwork<GM_Net, Coordinator, LearningNode> {
        typedef ::gm_protocol::GmLearningNetwork<network_t, coordinator_t, node_t> gm_learning_network_t;

        GM_Net(const set<source_id> &_hids, const string &_name, ContinuousQuery *_Q);
    };

    struct Coordinator : process {
        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef GM_Net network_t;

        proxy_map<node_proxy_t, node_t> proxy;

        /**
	        Protocol Stuff
        **/
        RNNLearner *global_learner;
        ContinuousQuery *Q;            // continuous query
        QueryState *query;             // current query state
        SafezoneFunction *safezone;    // the safe zone wrapper
        size_t k;                       // number of sites

        // index the nodes
        map<node_t *, size_t> nodeIndex;
        vector<node_t *> nodePtr;

        Coordinator(network_t *nw, ContinuousQuery *_Q);

        ~Coordinator();

        inline network_t *net() { return static_cast<network_t *>(host::net()); }

        inline const ProtocolConfig &cfg() const { return Q->config; }

        // Initialize the Learner and its' variables.
        void InitializeLearner();

        void SetupConnections() override;

        // load the warmup dataset
        void Warmup(arma::mat &batch, arma::mat &labels);

        // Ending the warmup of the network.
        void EndWarmup();

        // initialize a new round
        void StartRound();

        void FinishRound();

        void FinishRounds();

        // rebalance algorithm by Kamp
        void KampRebalance(node_t *n);

        // Printing and saving the accuracy.
        void Progress();

        // Getting the accuracy of the global learner.
        double GetAccuracy();

        // Get the communication statistics of experiment.
        vector<size_t> Statistics() const;

        // get the model of a node
        void FetchUpdates(node_t *n);

        // remote call on host violation
        oneway LocalViolation(sender<node_t> ctx);

        MatrixMessage HybridDrift(sender<node_t> ctx, IntNum rows, IntNum cols);

        MatrixMessage VirtualDrift(sender<node_t> ctx, IntNum rows);

        oneway RealDrift(sender<node_t> ctx, IntNum cols);

        set<node_t *> B;                     // initialized by local_violation(),
        // updated by rebalancing algo

        set<node_t *> Bcompl;             // complement of B, updated by rebalancing algo

        vector<arma::mat> Mean;          // Used to compute the mean model
        size_t num_violations;           // Number of violations in the same round (for rebalancing)

        int cnt;                         // Helping counter.

        // statistics
        size_t num_rounds;                 // total number of rounds
        size_t num_subrounds;             // total number of subrounds
        size_t sz_sent;                     // total safe zones sent
        size_t total_updates;             // number of stream updates received

    };

    struct coord_proxy : remote_proxy<Coordinator> {
        using coordinator_t = Coordinator;
        REMOTE_METHOD(coordinator_t, LocalViolation);
        REMOTE_METHOD(coordinator_t, HybridDrift);
        REMOTE_METHOD(coordinator_t, VirtualDrift);
        REMOTE_METHOD(coordinator_t, RealDrift);

        coord_proxy(process *c) : remote_proxy<coordinator_t>(c) {}
    };


/** This is a site implementation for
 * the classic Geometric Method protocol.
 * */
    struct LearningNode : local_site {

        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef GM_Net network_t;
        typedef coord_proxy coord_proxy_t;
        typedef ContinuousQuery continuous_query_t;

        continuous_query_t *Q;                // The query management object
        Safezone szone;                     // The safezone object
        RNNLearner *_learner;               // The learning algorithm

        vector<arma::mat> drift;            // The drift of the node

        int num_sites;                      // Number of sites

        size_t datapoints_seen;             // Number of points the node has seen since the last synchronization
        coord_proxy_t coord;                // The proxy of the coordinator/hub

        LearningNode(network_t *net, source_id hid, continuous_query_t *_Q)
                : local_site(net, hid), Q(_Q), coord(this) {
            coord <<= net->hub;
            InitializeLearner();
        };

        inline const ProtocolConfig &cfg() const { return Q->config; }

        void InitializeLearner();

        void SetupConnections();

        void UpdateDrift(vector<arma::mat *> &params);

        void UpdateStream(arma::mat &batch, arma::mat &labels);

        oneway SetHStaticVariables(const PModelState &SHParams);

        //
        // Remote methods
        //

        // Called at the start of a round
        oneway Reset(const Safezone &newsz);

        // Transfer data to the coordinator
        ModelState GetDrift();

        // Set the drift vector (for rebalancing)
        void SetDrift(ModelState mdl);

    };

    struct LearningNodeProxy : remote_proxy<LearningNode> {
        typedef LearningNode node_t;
        REMOTE_METHOD(node_t, Reset);
        REMOTE_METHOD(node_t, GetDrift);
        REMOTE_METHOD(node_t, SetDrift);
        REMOTE_METHOD(node_t, SetHStaticVariables);

        learning_node_proxy(process
        *p) :

        remote_proxy<node_t>(p) {}
    };


} // gm_protocol

namespace dds {
    template<>
    inline size_t byte_size<gm_protocol::LearningNode *>(
            gm_protocol::LearningNode *const &) { return 4; }
}

#endif