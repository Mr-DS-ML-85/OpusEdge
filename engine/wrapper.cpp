#include <Python.h>
#include "include/opusedge/core/signal.h"
#include "include/opusedge/primitives/delta_ar.h"
#include "include/opusedge/primitives/selkv.h"
#include "include/opusedge/primitives/smsa.h"
#include "include/opusedge/primitives/delta_rank.h"
#include "include/opusedge/primitives/head_gate.h"
#include "include/opusedge/primitives/state_compress.h"
#include "include/opusedge/primitives/composite.h"
#include "include/opusedge/primitives/pareto.h"
#include "include/opusedge/primitives/cal.h"
#include "include/opusedge/primitives/rcal.h"
#include "include/opusedge/primitives/gakv.h"
#include "include/opusedge/primitives/ndpa.h"
#include "include/opusedge/primitives/mpsr.h"
#include "include/opusedge/primitives/ebar.h"
#include "include/opusedge/primitives/ssr.h"
#include "include/opusedge/primitives/ipss.h"
#include <cstring>
#include <vector>

using namespace opusedge;

// ── Helpers ──
template<typename T>
static PyObject* vec_to_list(const Eigen::Matrix<T, -1, 1>& v) {
    PyObject* lst = PyList_New(v.size());
    for (int i = 0; i < v.size(); ++i)
        PyList_SET_ITEM(lst, i, PyFloat_FromDouble(v[i]));
    return lst;
}

static std::vector<double> py_to_vec(PyObject* ps) {
    Py_ssize_t n = PyObject_Length(ps);
    std::vector<double> buf(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    return buf;
}

// ── proxy_delta(scores: list[list[float]]) -> list[float] ──
static PyObject* py_proxy_delta(PyObject* self, PyObject* args) {
    PyObject* py_layers;
    if (!PyArg_ParseTuple(args, "O", &py_layers))
        return NULL;
    Py_ssize_t n_layers = PyObject_Length(py_layers);
    std::vector<MatrixXf> states;
    for (Py_ssize_t l = 0; l < n_layers; ++l) {
        PyObject* py_mat = PySequence_GetItem(py_layers, l);
        Py_ssize_t rows = PyObject_Length(py_mat);
        Py_ssize_t cols = 0;
        if (rows > 0) {
            PyObject* first_row = PySequence_GetItem(py_mat, 0);
            cols = PyObject_Length(first_row);
            Py_DECREF(first_row);
        }
        MatrixXf m(rows, cols);
        for (Py_ssize_t i = 0; i < rows; ++i) {
            PyObject* py_row = PySequence_GetItem(py_mat, i);
            for (Py_ssize_t j = 0; j < cols; ++j) {
                PyObject* val = PySequence_GetItem(py_row, j);
                m(i, j) = PyFloat_AsDouble(val);
                Py_DECREF(val);
            }
            Py_DECREF(py_row);
        }
        states.push_back(std::move(m));
        Py_DECREF(py_mat);
    }
    VectorXf result = SignalExtractor::proxy_delta(states);
    return vec_to_list(result);
}

// ── spearman(x: list[float], y: list[float]) -> float ──
static PyObject* py_spearman(PyObject* self, PyObject* args) {
    PyObject* px; PyObject* py;
    if (!PyArg_ParseTuple(args, "OO", &px, &py))
        return NULL;
    Py_ssize_t n = PyObject_Length(px);
    double* bx = new double[n]; double* by = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* ix = PySequence_GetItem(px, i);
        PyObject* iy = PySequence_GetItem(py, i);
        bx[i] = PyFloat_AsDouble(ix);
        by[i] = PyFloat_AsDouble(iy);
        Py_DECREF(ix); Py_DECREF(iy);
    }
    Eigen::Map<VectorXf> xv(bx, n), yv(by, n);
    double r = SignalExtractor::compute_spearman(xv, yv);
    delete[] bx; delete[] by;
    return PyFloat_FromDouble(r);
}

// ── normalize(signal: list[float]) -> list[float] ──
static PyObject* py_normalize(PyObject* self, PyObject* args) {
    PyObject* ps;
    if (!PyArg_ParseTuple(args, "O", &ps))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* buf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    Eigen::Map<VectorXf> s(buf, n);
    VectorXf r = SignalExtractor::normalize_importance(s);
    delete[] buf;
    return vec_to_list(r);
}

// ── sact_transmute(scores: list[float], residual: float) -> list[float] ──
static PyObject* py_sact(PyObject* self, PyObject* args) {
    PyObject* ps; double residual;
    if (!PyArg_ParseTuple(args, "Od", &ps, &residual))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* buf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    Eigen::Map<VectorXf> s(buf, n);
    VectorXf r = SignalExtractor::sact_transmute(s, residual);
    delete[] buf;
    return vec_to_list(r);
}

// ── delta_ar_indices(scores: list[float], top_k: int) -> list[list[float]] ──
static PyObject* py_delta_ar_indices(PyObject* self, PyObject* args) {
    PyObject* ps; int top_k;
    if (!PyArg_ParseTuple(args, "Oi", &ps, &top_k))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* buf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    Eigen::Map<VectorXf> d(buf, n);
    MatrixXf idx = build_delta_ar_indices(d, top_k);
    delete[] buf;
    PyObject* lst = PyList_New(n);
    for (int i = 0; i < n; ++i) {
        PyObject* row = PyList_New(top_k);
        for (int j = 0; j < top_k; ++j)
            PyList_SET_ITEM(row, j, PyLong_FromLong((long)idx(i, j)));
        PyList_SET_ITEM(lst, i, row);
    }
    return lst;
}

// ── delta_ar_flops(S: int, K: int) -> float ──
static PyObject* py_delta_ar_flops(PyObject* self, PyObject* args) {
    int S, K;
    if (!PyArg_ParseTuple(args, "ii", &S, &K))
        return NULL;
    return PyFloat_FromDouble(delta_ar_flops(S, K).reduction);
}

// ── ebdar(scores: list[float], mask: list[list[float]], beta: float) -> tuple ──
static PyObject* py_ebdar(PyObject* self, PyObject* args) {
    PyObject* ps, *pm; double beta;
    if (!PyArg_ParseTuple(args, "OOd", &ps, &pm, &beta))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* sbuf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        sbuf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    MatrixXf mask(n, n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* row = PySequence_GetItem(pm, i);
        for (Py_ssize_t j = 0; j < n; ++j) {
            PyObject* v = PySequence_GetItem(row, j);
            mask(i, j) = PyFloat_AsDouble(v);
            Py_DECREF(v);
        }
        Py_DECREF(row);
    }
    Eigen::Map<VectorXf> s(sbuf, n);
    auto [E, O] = ebdar(s, mask, beta);
    delete[] sbuf;
    return PyTuple_Pack(2, vec_to_list(E), vec_to_list(O));
}

// ── selkv_evict(scores: list[float], ratio: float) -> tuple ──
static PyObject* py_selkv_evict(PyObject* self, PyObject* args) {
    PyObject* ps; double ratio;
    if (!PyArg_ParseTuple(args, "Od", &ps, &ratio))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* buf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    Eigen::Map<VectorXf> d(buf, n);
    auto ev = SelKV::evict(d, ratio, n);
    delete[] buf;
    PyObject* ret = PyList_New(ev.retained_indices.size());
    for (size_t i = 0; i < ev.retained_indices.size(); ++i)
        PyList_SET_ITEM(ret, i, PyLong_FromLong(ev.retained_indices[i]));
    PyObject* evi = PyList_New(ev.evicted_indices.size());
    for (size_t i = 0; i < ev.evicted_indices.size(); ++i)
        PyList_SET_ITEM(evi, i, PyLong_FromLong(ev.evicted_indices[i]));
    return PyTuple_Pack(2, ret, evi);
}

// ── selkv_quality_ratio(scores: list[float], ratio: float) -> float ──
static PyObject* py_selkv_quality_ratio(PyObject* self, PyObject* args) {
    PyObject* ps; double ratio;
    if (!PyArg_ParseTuple(args, "Od", &ps, &ratio))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* buf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    Eigen::Map<VectorXf> d(buf, n);
    double r = SelKV::quality_ratio(d, ratio);
    delete[] buf;
    return PyFloat_FromDouble(r);
}

// ── smsa_analyze(seq_len: int, window: int) -> tuple ──
static PyObject* py_smsa_analyze(PyObject* self, PyObject* args) {
    int seq_len, window;
    if (!PyArg_ParseTuple(args, "ii", &seq_len, &window))
        return NULL;
    SMSA smsa(window);
    auto r = smsa.analyze(seq_len);
    return Py_BuildValue("(ddi)", r.speedup, r.memory_savings, r.effective_window);
}

// ── head_active(delta_t: float, n_heads: int = 32) -> int ──
static PyObject* py_head_active(PyObject* self, PyObject* args) {
    double dt; int n_heads = 32;
    if (!PyArg_ParseTuple(args, "d|i", &dt, &n_heads))
        return NULL;
    HeadGateConfig cfg; cfg.n_heads = n_heads;
    HeadGate hg(cfg);
    return PyLong_FromLong(hg.active_heads(dt));
}

// ── head_flop_reduction(deltas: list[float]) -> float ──
static PyObject* py_head_flop_reduction(PyObject* self, PyObject* args) {
    PyObject* ps;
    if (!PyArg_ParseTuple(args, "O", &ps))
        return NULL;
    Py_ssize_t n = PyObject_Length(ps);
    double* buf = new double[n];
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PySequence_GetItem(ps, i);
        buf[i] = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    HeadGateConfig cfg; cfg.n_heads = 32;
    HeadGate hg(cfg);
    Eigen::Map<VectorXf> d(buf, n);
    auto r = hg.analyze(d);
    delete[] buf;
    return PyFloat_FromDouble(r.flop_reduction_pct);
}

// ── state_keep_ratio(delta_t: float) -> float ──
static PyObject* py_state_keep_ratio(PyObject* self, PyObject* args) {
    double dt;
    if (!PyArg_ParseTuple(args, "d", &dt))
        return NULL;
    StateCompress sc;
    return PyFloat_FromDouble(sc.keep_ratio(dt));
}

// ── composite_analyze(seq_len, n_heads, head_dim, hidden_dim) -> tuple ──
static PyObject* py_composite_analyze(PyObject* self, PyObject* args) {
    int seq_len, n_heads, head_dim, hidden_dim;
    if (!PyArg_ParseTuple(args, "iiii", &seq_len, &n_heads, &head_dim, &hidden_dim))
        return NULL;
    CompositeConfig cfg;
    Composite comp(cfg);
    auto r = comp.analyze(seq_len, n_heads, head_dim, hidden_dim);
    return Py_BuildValue("(dd)", r.flop_reduction_pct, r.memory_savings_pct);
}

// ── gakv_analyze(deltas: list[float], ir: list[float]) -> tuple ──
static PyObject* py_gakv_analyze(PyObject* self, PyObject* args) {
    PyObject *pd, *pir;
    if (!PyArg_ParseTuple(args, "OO", &pd, &pir))
        return NULL;
    auto dv = py_to_vec(pd);
    auto irv = py_to_vec(pir);
    GAKV g;
    auto r = g.analyze(dv, irv);
    PyObject* scores = PyList_New(r.composite_scores.size());
    for (size_t i = 0; i < r.composite_scores.size(); ++i)
        PyList_SET_ITEM(scores, i, PyFloat_FromDouble(r.composite_scores[i]));
    PyObject* ev = PyList_New(r.evicted_indices.size());
    for (size_t i = 0; i < r.evicted_indices.size(); ++i)
        PyList_SET_ITEM(ev, i, PyLong_FromLong(r.evicted_indices[i]));
    PyObject* rt = PyList_New(r.retained_indices.size());
    for (size_t i = 0; i < r.retained_indices.size(); ++i)
        PyList_SET_ITEM(rt, i, PyLong_FromLong(r.retained_indices[i]));
    return PyTuple_Pack(3, scores, rt, ev);
}

// ── rgakv_analyze(deltas: list[float], ir: list[float], cal_mod: float) -> tuple ──
static PyObject* py_rgakv_analyze(PyObject* self, PyObject* args) {
    PyObject *pd, *pir; double cal_mod;
    if (!PyArg_ParseTuple(args, "OOd", &pd, &pir, &cal_mod))
        return NULL;
    auto dv = py_to_vec(pd);
    auto irv = py_to_vec(pir);
    GAKV g;
    auto r = g.rgakv_analyze(dv, irv, cal_mod);
    PyObject* scores = PyList_New(r.composite_scores.size());
    for (size_t i = 0; i < r.composite_scores.size(); ++i)
        PyList_SET_ITEM(scores, i, PyFloat_FromDouble(r.composite_scores[i]));
    PyObject* ev = PyList_New(r.evicted_indices.size());
    for (size_t i = 0; i < r.evicted_indices.size(); ++i)
        PyList_SET_ITEM(ev, i, PyLong_FromLong(r.evicted_indices[i]));
    PyObject* rt = PyList_New(r.retained_indices.size());
    for (size_t i = 0; i < r.retained_indices.size(); ++i)
        PyList_SET_ITEM(rt, i, PyLong_FromLong(r.retained_indices[i]));
    return PyTuple_Pack(3, scores, rt, ev);
}

// ── ndpa_rectify(proxy_delta: list[float], attn_scores: list[float]) -> tuple ──
static PyObject* py_ndpa_rectify(PyObject* self, PyObject* args) {
    PyObject *pd, *pa;
    if (!PyArg_ParseTuple(args, "OO", &pd, &pa))
        return NULL;
    auto dv = py_to_vec(pd);
    auto av = py_to_vec(pa);
    NDPA ndpa;
    auto r = ndpa.analyze(dv, av);
    PyObject* rect = PyList_New(r.rectified_delta.size());
    for (size_t i = 0; i < r.rectified_delta.size(); ++i)
        PyList_SET_ITEM(rect, i, PyFloat_FromDouble(r.rectified_delta[i]));
    return Py_BuildValue("(Odi)", rect, r.gamma, r.active ? 1 : 0);
}

// ── mpsr_project(evicted_scores: list[float], state_dim: float) -> tuple ──
static PyObject* py_mpsr_project(PyObject* self, PyObject* args) {
    PyObject* ps; double state_dim;
    if (!PyArg_ParseTuple(args, "Od", &ps, &state_dim))
        return NULL;
    auto ev = py_to_vec(ps);
    MPSR m;
    auto r = m.project(ev, state_dim);
    PyObject* proj = PyList_New(r.projected_state.size());
    for (size_t i = 0; i < r.projected_state.size(); ++i)
        PyList_SET_ITEM(proj, i, PyFloat_FromDouble(r.projected_state[i]));
    return Py_BuildValue("(Odd)", proj, r.compression_ratio, r.energy_retained);
}

// ── mpsr_sact(evicted_scores: list[float], state_dim: float, residual: float) -> tuple ──
static PyObject* py_mpsr_sact(PyObject* self, PyObject* args) {
    PyObject* ps; double state_dim, residual;
    if (!PyArg_ParseTuple(args, "Odd", &ps, &state_dim, &residual))
        return NULL;
    auto ev = py_to_vec(ps);
    MPSR m;
    auto r = m.project_sact(ev, state_dim, residual);
    PyObject* proj = PyList_New(r.projected_state.size());
    for (size_t i = 0; i < r.projected_state.size(); ++i)
        PyList_SET_ITEM(proj, i, PyFloat_FromDouble(r.projected_state[i]));
    return Py_BuildValue("(Odd)", proj, r.compression_ratio, r.energy_retained);
}

// ── ebar_analyze(entropies: list[float]) -> tuple ──
static PyObject* py_ebar_analyze(PyObject* self, PyObject* args) {
    PyObject* ps;
    if (!PyArg_ParseTuple(args, "O", &ps))
        return NULL;
    auto ev = py_to_vec(ps);
    EBAR e;
    auto r = e.analyze(ev);
    PyObject* compute = PyList_New(r.compute_per_step.size());
    PyObject* buffer = PyList_New(r.entropy_buffer.size());
    for (size_t i = 0; i < r.compute_per_step.size(); ++i) {
        PyList_SET_ITEM(compute, i, PyFloat_FromDouble(r.compute_per_step[i]));
        PyList_SET_ITEM(buffer, i, PyFloat_FromDouble(r.entropy_buffer[i]));
    }
    return Py_BuildValue("(OOd)", compute, buffer, r.total_compute_savings);
}

// ── ebar_entropy(log_probs: list[float]) -> list[float] ──
static PyObject* py_ebar_entropy(PyObject* self, PyObject* args) {
    PyObject* ps;
    if (!PyArg_ParseTuple(args, "O", &ps))
        return NULL;
    auto ev = py_to_vec(ps);
    auto ent = EBAR::compute_shannon_entropy(ev);
    PyObject* lst = PyList_New(ent.size());
    for (size_t i = 0; i < ent.size(); ++i)
        PyList_SET_ITEM(lst, i, PyFloat_FromDouble(ent[i]));
    return lst;
}

// ── ssr_analyze(singular_values: list[float], layer_entropy: float) -> tuple ──
static PyObject* py_ssr_analyze(PyObject* self, PyObject* args) {
    PyObject* ps; double entropy;
    if (!PyArg_ParseTuple(args, "Od", &ps, &entropy))
        return NULL;
    auto sv = py_to_vec(ps);
    SSR s;
    auto r = s.analyze(sv, entropy);
    PyObject* vals = PyList_New(r.thresholded_values.size());
    for (size_t i = 0; i < r.thresholded_values.size(); ++i)
        PyList_SET_ITEM(vals, i, PyFloat_FromDouble(r.thresholded_values[i]));
    return Py_BuildValue("(Oidi)", vals, r.preserved_count, r.preserved_fraction, r.compression_ratio);
}

// ── ssr_casp(singular_values: list[float], curvature: float) -> tuple ──
static PyObject* py_ssr_casp(PyObject* self, PyObject* args) {
    PyObject* ps; double curvature;
    if (!PyArg_ParseTuple(args, "Od", &ps, &curvature))
        return NULL;
    auto sv = py_to_vec(ps);
    SSR s;
    auto r = s.casp_analyze(sv, curvature);
    PyObject* vals = PyList_New(r.thresholded_values.size());
    for (size_t i = 0; i < r.thresholded_values.size(); ++i)
        PyList_SET_ITEM(vals, i, PyFloat_FromDouble(r.thresholded_values[i]));
    return Py_BuildValue("(Oidi)", vals, r.preserved_count, r.preserved_fraction, r.compression_ratio);
}

// ── ipss_analyze(head_variances: list[float]) -> tuple ──
static PyObject* py_ipss_analyze(PyObject* self, PyObject* args) {
    PyObject* ps;
    if (!PyArg_ParseTuple(args, "O", &ps))
        return NULL;
    auto hv = py_to_vec(ps);
    IPSS ipss;
    auto r = ipss.analyze_simple(hv);
    PyObject* active = PyList_New(r.active_heads.size());
    for (size_t i = 0; i < r.active_heads.size(); ++i)
        PyList_SET_ITEM(active, i, PyLong_FromLong(r.active_heads[i]));
    PyObject* sal = PyList_New(r.salience_values.size());
    for (size_t i = 0; i < r.salience_values.size(); ++i)
        PyList_SET_ITEM(sal, i, PyFloat_FromDouble(r.salience_values[i]));
    return Py_BuildValue("(OOdi)", active, sal, r.flop_reduction, r.n_active);
}

// ── cal_classify(task: str, base_threshold: float) -> tuple ──
static PyObject* py_cal_classify(PyObject* self, PyObject* args) {
    const char* task; double base;
    if (!PyArg_ParseTuple(args, "sd", &task, &base))
        return NULL;
    opusedge::CAL cal;
    auto r = cal.classify(task);
    double eff = cal.effective_threshold(task);
    return Py_BuildValue("(sidd)", r.name.c_str(), static_cast<int>(r.tier),
                         r.rigidity, eff);
}

// ── cal_rigidity(task: str) -> float ──
static PyObject* py_cal_rigidity(PyObject* self, PyObject* args) {
    const char* task;
    if (!PyArg_ParseTuple(args, "s", &task))
        return NULL;
    opusedge::CAL cal;
    return PyFloat_FromDouble(cal.rigidity_of(task));
}

// ── rcal_classify(task: str, base_threshold: float) -> tuple ──
static PyObject* py_rcal_classify(PyObject* self, PyObject* args) {
    const char* task; double base;
    if (!PyArg_ParseTuple(args, "sd", &task, &base))
        return NULL;
    RCAL rcal;
    auto r = rcal.classify(task, base);
    return Py_BuildValue("(ssdd)", r.tier.name.c_str(), r.tier.name.c_str(),
                         r.effective_threshold, r.confidence);
}

// ── rcal_modulate(task: str, base: float) -> float ──
static PyObject* py_rcal_modulate(PyObject* self, PyObject* args) {
    const char* task; double base;
    if (!PyArg_ParseTuple(args, "sd", &task, &base))
        return NULL;
    RCAL rcal;
    return PyFloat_FromDouble(rcal.modulate_threshold(task, base));
}

// ── rcal_eviction_cap(task: str) -> float ──
static PyObject* py_rcal_eviction_cap(PyObject* self, PyObject* args) {
    const char* task;
    if (!PyArg_ParseTuple(args, "s", &task))
        return NULL;
    RCAL rcal;
    return PyFloat_FromDouble(rcal.modulated_eviction_cap(task));
}

// ── pareto_sweep(deltas: list[float], seq_len: int) -> list ──
static PyObject* py_pareto_sweep(PyObject* self, PyObject* args) {
    PyObject* ps; int seq_len;
    if (!PyArg_ParseTuple(args, "Oi", &ps, &seq_len))
        return NULL;
    auto dv = py_to_vec(ps);
    int n = dv.size();
    double* buf = new double[n];
    for (int i = 0; i < n; ++i) buf[i] = dv[i];
    Eigen::Map<VectorXf> d(buf, n);
    auto pts = ParetoFrontier::sweep(d, seq_len);
    delete[] buf;
    PyObject* lst = PyList_New(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        PyObject* pt = Py_BuildValue("(diidddd)", pts[i].eviction_ratio, pts[i].window_size,
                                      pts[i].rank_fraction, pts[i].channel_keep,
                                      pts[i].ppl, pts[i].flops, pts[i].mem_bytes, pts[i].delta_ppl);
        PyList_SET_ITEM(lst, i, pt);
    }
    return lst;
}

// ── Method table ──
static PyMethodDef OpusEdgeMethods[] = {
    {"proxy_delta",          py_proxy_delta,          METH_VARARGS, "Proxy-Δ signal from hidden states across layers"},
    {"spearman",             py_spearman,             METH_VARARGS, "Spearman rank correlation"},
    {"normalize",            py_normalize,            METH_VARARGS, "Normalize importance scores"},
    {"sact_transmute",       py_sact,                 METH_VARARGS, "SACT transmutation"},
    {"delta_ar_indices",     py_delta_ar_indices,     METH_VARARGS, "Build Delta-AR routing indices (O(S log K))"},
    {"delta_ar_flops",       py_delta_ar_flops,       METH_VARARGS, "Delta-AR FLOP reduction"},
    {"ebdar",                py_ebdar,                METH_VARARGS, "EB-DAR (Eq 10a/10b) with β·E injection"},
    {"selkv_evict",          py_selkv_evict,          METH_VARARGS, "SelKV eviction by Δ-importance"},
    {"selkv_quality_ratio",  py_selkv_quality_ratio,  METH_VARARGS, "SelKV quality ratio vs random eviction"},
    {"smsa_analyze",         py_smsa_analyze,         METH_VARARGS, "SMSA speedup / memory analysis"},
    {"head_active",          py_head_active,          METH_VARARGS, "Active heads count at given Δ threshold"},
    {"head_flop_reduction",  py_head_flop_reduction,  METH_VARARGS, "FLOP reduction from head deactivation"},
    {"state_keep_ratio",     py_state_keep_ratio,     METH_VARARGS, "State compression keep ratio"},
    {"composite_analyze",    py_composite_analyze,    METH_VARARGS, "Composite OpusEdge pipeline analysis"},
    {"gakv_analyze",         py_gakv_analyze,         METH_VARARGS, "GAKV: Gating-Aware KV composite scoring"},
    {"rgakv_analyze",        py_rgakv_analyze,        METH_VARARGS, "R-GAKV: GAKV with CAL-modulated thresholds"},
    {"ndpa_rectify",         py_ndpa_rectify,         METH_VARARGS, "NDPA: Neural Delta-Phase Alignment rectifier"},
    {"mpsr_project",         py_mpsr_project,         METH_VARARGS, "MPSR: Manifold-Preserving State Recycling"},
    {"mpsr_sact",            py_mpsr_sact,            METH_VARARGS, "MPSR with SACT residual injection"},
    {"ebar_analyze",         py_ebar_analyze,         METH_VARARGS, "EB-AR: Entropy-Buffered Autoregression"},
    {"ebar_entropy",         py_ebar_entropy,         METH_VARARGS, "Shannon entropy from log-probs"},
    {"ssr_analyze",          py_ssr_analyze,          METH_VARARGS, "SSR: Soft Spectral Relaxation"},
    {"ssr_casp",             py_ssr_casp,             METH_VARARGS, "CASP: Curvature-Adaptive Spectral Projection"},
    {"ipss_analyze",         py_ipss_analyze,         METH_VARARGS, "IPSS: Info-Preserving Salience Smoothing"},
    {"rcal_classify",        py_rcal_classify,        METH_VARARGS, "R-CAL: classify task into integrity tier (hybrid)"},
    {"rcal_modulate",        py_rcal_modulate,        METH_VARARGS, "R-CAL: modulate threshold by task rigidity"},
    {"rcal_eviction_cap",    py_rcal_eviction_cap,    METH_VARARGS, "R-CAL: eviction cap for task"},
    {"cal_classify",         py_cal_classify,         METH_VARARGS, "CAL: classify task into integrity tier (dense/MoE)"},
    {"cal_rigidity",         py_cal_rigidity,         METH_VARARGS, "CAL: task rigidity value"},
    {"pareto_sweep",         py_pareto_sweep,         METH_VARARGS, "Pareto frontier sweep over config space"},
    {NULL, NULL, 0, NULL}
};

// Compiled as opusedge_cpp._core so the Python facade at
// opusedge_cpp/__init__.py can `from ._core import *` cleanly.
static struct PyModuleDef OpusEdgeModule = {
    PyModuleDef_HEAD_INIT,
    "opusedge_cpp._core",
    "OpusEdge C++ core — native primitives + policy framework",
    -1,
    OpusEdgeMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&OpusEdgeModule);
}
