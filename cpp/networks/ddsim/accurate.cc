
#include <iomanip>
#include "binc.hh"
#include "accurate.hh"
#include "cfgfile.hh"

using namespace dds;


namespace dds {

    template<>
    data_source_statistics *component_type<data_source_statistics>::create(const Json::Value &) {
        return new data_source_statistics();
    }

}

dds::component_type<data_source_statistics> data_source_statistics::comp_type("data_source_statistics");


data_source_statistics::data_source_statistics()
        : component("datasrc_stat") {
    on(START_RECORD, [&]() { process(CTX.stream_record()); });
    on(RESULTS, [&]() { finish(); });

    for (auto sid : CTX.metadata().stream_ids())
        for (auto hid : CTX.metadata().source_ids()) {
            char buf[32];
            sprintf(buf, "LSSize_s%d_h%d", sid, hid);
            col_t *c = new col_t(buf, "%d", 0);
            lssize[local_stream_id{sid, hid}] = c;
            CTX.timeseries.add(*c);
        }

    for (auto sid : CTX.metadata().stream_ids()) {
        char buf[32];
        sprintf(buf, "SSize_s%d", sid);
        col_t *c = new col_t(buf, "%d", 0);
        ssize[sid] = c;
        CTX.timeseries.add(*c);
    }

    for (auto hid : CTX.metadata().source_ids()) {
        char buf[32];
        sprintf(buf, "SSize_h%d", hid);
        col_t *c = new col_t(buf, "%d", 0);
        hsize[hid] = c;
        CTX.timeseries.add(*c);
    }

}

void data_source_statistics::process(const dds_record &rec) {
    if (scount == 0) ts = rec.ts;
    te = rec.ts;
    sids.insert(rec.sid);
    hids.insert(rec.hid);

    stream_size[rec.sid]++;
    //std::cout << rec << std::endl;
    lshist[rec.local_stream()]++;

    lssize[rec.local_stream()]->value() += rec.upd;
    ssize[rec.sid]->value() += rec.upd;
    hsize[rec.hid]->value() += rec.upd;
    scount++;
}


void data_source_statistics::finish() {
    report(std::cout);
}


data_source_statistics::~data_source_statistics() {
    // clear timeseries columns
    for (auto i : lssize)
        delete i.second;
    for (auto i : ssize)
        delete i.second;
    for (auto i : hsize)
        delete i.second;
}


void data_source_statistics::report(std::ostream &s) {
    using std::endl;
    using std::setw;

    s << "Stats:" << scount
      << " streams=" << sids.size()
      << " local hosts=" << hids.size() << endl;

    const int CW = 9;
    const int NW = 10;

    // header
    s << setw(CW) << "Stream:";
    for (auto sid : sids)
        s << setw(NW) << sid;
    s << endl;

    for (auto hid: hids) {
        s << "host " << setw(3) << hid << ":";
        for (auto sid: sids) {
            auto stream_len = lshist[local_stream_id{sid, hid}];

            local_stream_stats.sid = sid;
            local_stream_stats.hid = hid;
            local_stream_stats.stream_len = stream_len;
            local_stream_stats.emit_row();

            s << setw(NW) << stream_len;
        }
        s << endl;
    }

    for (auto sid = stream_size.begin(); sid != stream_size.end(); sid++) {
        s << "stream[" << sid->first << "]=" << sid->second << endl;
    }
}


local_stream_stats_t dds::local_stream_stats;


//
//////////////////////////////////////////////
//




selfjoin_exact_method::selfjoin_exact_method(const string &n, stream_id sid)
        : query_method(n, self_join(sid)) {
    on(START_STREAM, [&]() { process_warmup(CTX.warmup); });
    on(START_RECORD, [&]() { process_record(CTX.stream_record()); });
    on(END_STREAM, [&]() { finish(); });
}

void selfjoin_exact_method::process_warmup(const buffered_dataset &wset) {
    for (auto &&rec: wset) {
        process_record(rec);
    }
}


void selfjoin_exact_method::process_record(const dds_record &rec) {
    if (rec.sid == Q.operand(0)) {

        long &x = histogram.get_counter(rec.key);
        curest += (2 * x + rec.upd) * rec.upd;
        x += rec.upd;

    }
}

void selfjoin_exact_method::finish() {
    cout << "selfjoin(" << Q.operand(0) << ")=" << curest << endl;
}



//
//////////////////////////////////////////////
//


twoway_join_exact_method::twoway_join_exact_method(const string &n, stream_id s1, stream_id s2)
        : query_method(n, join(s1, s2)) {
    on(START_STREAM, [&]() { process_warmup(CTX.warmup); });
    on(START_RECORD, [=]() {
        process_record(CTX.stream_record());
    });
    on(END_STREAM, [=]() {
        finish();
    });
}

// update goes in h1
void twoway_join_exact_method::
dojoin(histogram &h1, histogram &h2, const dds_record &rec) {
    auto &x = h1.get_counter(rec.key);
    auto &y = h2.get_counter(rec.key);

    x += rec.upd;
    curest += rec.upd * y;
}

void twoway_join_exact_method::process_warmup(const buffered_dataset &wset) {
    for (auto &&rec: wset) {
        process_record(rec);
    }
}


void twoway_join_exact_method::process_record(const dds_record &rec) {
    if (rec.sid == Q.operand(0)) {
        dojoin(hist1, hist2, rec);
    } else if (rec.sid == Q.operand(1)) {
        dojoin(hist2, hist1, rec);
    }
}

void twoway_join_exact_method::finish() {
    cout << "2wayjoin("
         << Q.operand(0) << "," << Q.operand(1) << ")="
         << curest << endl;
}


//
//////////////////////////////////////////////
//


agms_sketch_updater::agms_sketch_updater(stream_id _sid, agms::projection proj)
        : sid(_sid), isk(proj) {
    on(START_STREAM, [&]() {
        for (auto &&rec : CTX.warmup) {
            if (rec.sid == sid) {
                isk.update(rec.key, rec.upd);
            }
        }
        emit(STREAM_SKETCH_INITIALIZED);
    });

    on(START_RECORD, [&]() {
        const dds_record &rec = CTX.stream_record();
        if (rec.sid == sid) {
            isk.update(rec.key, rec.upd);
            emit(STREAM_SKETCH_UPDATED);
        }
    });
}


factory<agms_sketch_updater, stream_id, agms::projection>
        dds::agms_sketch_updater_factory;


selfjoin_agms_method::selfjoin_agms_method(const string &n, stream_id sid,
                                           agms::depth_type D, size_t L)
        : selfjoin_agms_method(n, sid, agms::projection(D, L)) {}

selfjoin_agms_method::selfjoin_agms_method(const string &n, stream_id sid,
                                           const agms::projection &proj)
        : query_method(n, self_join(sid, proj.epsilon())) {
    using namespace agms;

    isk = &agms_sketch_updater_factory(sid, proj)->isk;

    on(STREAM_SKETCH_INITIALIZED, [&]() { initialize(); });
    on(STREAM_SKETCH_UPDATED, [&]() { process_record(); });
}


void selfjoin_agms_method::initialize() {
    if (isinit) return;
    curest = dot_est_with_inc(incstate, *isk);
    isinit = true;
}


void selfjoin_agms_method::process_record() {
    using namespace agms;

    if (CTX.stream_record().sid == Q.operand(0)) {
        curest = dot_est_inc(incstate, isk->delta);
    }
}


//
//////////////////////////////////////////////
//



twoway_join_agms_method::twoway_join_agms_method(const string &n, stream_id s1, stream_id s2,
                                                 agms::depth_type D, size_t L)
        : twoway_join_agms_method(n, s1, s2, agms::projection(D, L)) {}

twoway_join_agms_method::twoway_join_agms_method(const string &n, stream_id s1, stream_id s2,
                                                 const agms::projection &proj)
        : query_method(n, join(s1, s2, proj.epsilon())) {
    using namespace agms;
    isk1 = &agms_sketch_updater_factory(s1, proj)->isk;
    isk2 = &agms_sketch_updater_factory(s2, proj)->isk;

    on(STREAM_SKETCH_INITIALIZED, [&]() { initialize(); });
    on(STREAM_SKETCH_UPDATED, [&]() { process_record(); });
}

void twoway_join_agms_method::initialize() {
    if (isinit) return;
    curest = dot_est_with_inc(incstate, *isk1, *isk2);
    isinit = true;
}

void twoway_join_agms_method::process_record() {
    using namespace agms;
    if (CTX.stream_record().sid == Q.operand(0)) {
        curest = dot_est_inc(incstate, isk1->delta, *isk2);
    } else if (CTX.stream_record().sid == Q.operand(1)) {
        curest = dot_est_inc(incstate, *isk1, isk2->delta);
    }
}



//////////////////////////////////////////////
//
// Component types
//
//////////////////////////////////////////////

component_type<query_method> dds::exact_query_comptype("exact_query");
component_type<query_method> dds::agms_query_comptype("agms_query");

namespace dds {

    template<>
    query_method *component_type<query_method>::create(const Json::Value &js) {
        using binc::sprint;

        string _name = js["name"].asString();
        basic_stream_query Q = get_query(js);
        if (Q.type() != qtype::JOIN && Q.type() != qtype::SELFJOIN)
            throw std::runtime_error(sprint("Cannot handle query type ", Q.type()));


        if (name() == "exact_query") {
            if (Q.type() == qtype::SELFJOIN)
                return new selfjoin_exact_method(_name, Q.operand(0));
            else
                return new twoway_join_exact_method(_name, Q.operand(0), Q.operand(1));

        } else if (name() == "agms_query") {
            agms::projection proj = get_projection(js);
            if (Q.type() == qtype::SELFJOIN)
                return new selfjoin_agms_method(_name, Q.operand(0), proj);
            else
                return new twoway_join_agms_method(_name, Q.operand(0), Q.operand(1), proj);

        } else {
            assert(0);
            return NULL;
        }
    }


} //end namespace ddsim

