// FreeToken: Vulkan ggml ops -- the same op surface as ggml_ops.cu, executed as one-op
// ggml graphs on the vendored ggml-vulkan backend. Weights are uploaded once into
// persistent device buffers keyed by their torch storage; activations cross the host
// per call (fp32 compute), which keeps the graph small and the dtype rules simple.
// Torch tensors may live on CUDA (mixed CUDA/Vulkan serving) or on the CPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-vulkan.h"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>

#ifdef FREETOKEN_VK_CUDA
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <future>
#include <thread>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::TensorView;

namespace {

// ---------------------------------------------------------------- host staging

int64_t dtype_bytes(DLDataType dt) {
    if (dt.code == kDLFloat && dt.bits == 32) return 4;
    if (dt.code == kDLFloat && dt.bits == 16) return 2;
    if (dt.code == kDLBfloat && dt.bits == 16) return 2;
    if (dt.code == kDLInt && dt.bits == 32) return 4;
    if (dt.code == kDLUInt && dt.bits == 8) return 1;
    throw std::runtime_error("ggml vk: unsupported tensor dtype");
}

bool is_bf16(DLDataType dt) { return dt.code == kDLBfloat && dt.bits == 16; }
bool is_f16(DLDataType dt) { return dt.code == kDLFloat && dt.bits == 16; }

float bf16_to_f32(uint16_t v) {
    uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    // round to nearest even, the way torch does
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7fff + lsb;
    return uint16_t(u >> 16);
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000 | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = int32_t((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffff;
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) half++;
        return uint16_t(sign | half);
    }
    uint32_t half = mant >> 13;
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) {
        half++;
        if (half == 0x400) { half = 0; exp++; if (exp >= 31) return uint16_t(sign | 0x7c00); }
    }
    return uint16_t(sign | (uint32_t(exp) << 10) | half);
}

void cuda_sync(DLDevice d) {
#ifdef FREETOKEN_VK_CUDA
    if (d.device_type == kDLCUDA) {
        auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(d.device_type, d.device_id));
        cudaSetDevice(d.device_id);
        cudaStreamSynchronize(stream == nullptr ? cudaStreamLegacy : stream);
    }
#else
    (void) d;
#endif
}

// copy of a torch tensor into host memory, honoring strides: the packed weight views
// drop MMQ padding rows, so they are not contiguous
std::vector<uint8_t> to_host(TensorView t) {
    const int64_t esz = dtype_bytes(t.dtype());
    const int64_t n = t.numel() * esz;
    std::vector<uint8_t> h(n);
    if (n == 0) return h;
    const int nd = t.ndim();
    std::vector<int64_t> shape(nd), stride(nd);
    int64_t span = esz;
    for (int i = 0; i < nd; i++) {
        shape[i] = t.size(i);
        stride[i] = t.stride(i) * esz;  // torch DLPack always carries strides
        span += (shape[i] - 1) * stride[i];
    }
    if (nd > 0 && stride[nd - 1] != esz) throw std::runtime_error("ggml vk: innermost stride must be 1");
    std::vector<uint8_t> raw(span);
    cuda_sync(t.device());
#ifdef FREETOKEN_VK_CUDA
    if (t.device().device_type == kDLCUDA) {
        cudaSetDevice(t.device().device_id);
        auto err = cudaMemcpy(raw.data(), t.data_ptr(), span, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error(std::string("ggml vk D2H: ") + cudaGetErrorString(err));
    } else
#endif
    {
        std::memcpy(raw.data(), t.data_ptr(), span);
    }
    if (nd <= 1) {
        std::memcpy(h.data(), raw.data(), n);
        return h;
    }
    // gather the strided view into contiguous rows
    const int64_t run = shape[nd - 1] * esz;
    int64_t pos = 0;
    std::vector<int64_t> idx(nd - 1, 0);
    while (true) {
        int64_t off = 0;
        for (int i = 0; i + 1 < nd; i++) off += idx[i] * stride[i];
        std::memcpy(h.data() + pos, raw.data() + off, run);
        pos += run;
        if (pos >= n) break;
        int d = nd - 2;
        while (d >= 0 && ++idx[d] >= shape[d]) { idx[d] = 0; d--; }
        if (d < 0) break;
    }
    return h;
}

void from_host(TensorView out, const void * h, size_t n) {
    if (n == 0) return;
    cuda_sync(out.device());
#ifdef FREETOKEN_VK_CUDA
    if (out.device().device_type == kDLCUDA) {
        cudaSetDevice(out.device().device_id);
        auto err = cudaMemcpy(out.data_ptr(), h, n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error(std::string("ggml vk H2D: ") + cudaGetErrorString(err));
        return;
    }
#endif
    std::memcpy(out.data_ptr(), h, n);
}

// strided 1-D gather (the NVFP4 per-expert scale views are strided)
std::vector<float> scale_to_host(TensorView s, int64_t e) {
    cuda_sync(s.device());
    std::vector<float> out(e);
    const int64_t stride = s.stride(0);
    const uint8_t * base = static_cast<const uint8_t *>(s.data_ptr());
    for (int64_t i = 0; i < e; i++) {
        const void * p = base + i * stride * dtype_bytes(s.dtype());
        float v;
#ifdef FREETOKEN_VK_CUDA
        if (s.device().device_type == kDLCUDA) {
            cudaSetDevice(s.device().device_id);
            cudaMemcpy(&v, p, 4, cudaMemcpyDeviceToHost);
        } else
#endif
        {
            std::memcpy(&v, p, 4);
        }
        out[i] = v;
    }
    return out;
}

// activations upload as fp32; results download as fp32 and convert to the caller's dtype
std::vector<float> act_to_f32(TensorView x) {
    auto h = to_host(x);
    const int64_t n = x.numel();
    std::vector<float> f(n);
    if (is_bf16(x.dtype())) {
        auto p = reinterpret_cast<const uint16_t *>(h.data());
        for (int64_t i = 0; i < n; i++) f[i] = bf16_to_f32(p[i]);
    } else if (is_f16(x.dtype())) {
        auto p = reinterpret_cast<const uint16_t *>(h.data());
        for (int64_t i = 0; i < n; i++) f[i] = f16_to_f32(p[i]);
    } else {
        std::memcpy(f.data(), h.data(), n * 4);
    }
    return f;
}

// ---------------------------------------------------------------- backend state

struct WeightKey {
    uint64_t ptr;
    int64_t nbytes;
    int type;
    int64_t ne[4];
    int64_t nb[4];
    bool operator==(const WeightKey & o) const {
        return ptr == o.ptr && nbytes == o.nbytes && type == o.type &&
               ne[0] == o.ne[0] && ne[1] == o.ne[1] && ne[2] == o.ne[2] && ne[3] == o.ne[3] &&
               nb[0] == o.nb[0] && nb[1] == o.nb[1] && nb[2] == o.nb[2] && nb[3] == o.nb[3];
    }
};

struct WeightKeyHash {
    size_t operator()(const WeightKey & k) const {
        size_t h = std::hash<uint64_t>()(k.ptr);
        h = h * 31 + std::hash<int64_t>()(k.nbytes);
        h = h * 31 + std::hash<int>()(k.type);
        for (int i = 0; i < 4; i++) h = h * 31 + std::hash<int64_t>()(k.ne[i]);
        for (int i = 0; i < 4; i++) h = h * 31 + std::hash<int64_t>()(k.nb[i]);
        return h;
    }
};

struct VkState {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    // one gallocr per graph size: routing changes the node count per call, and a
    // shared gallocr would reallocate its buffers on every change
    std::unordered_map<int, ggml_gallocr_t> gallocrs;
    ggml_context * wctx = nullptr;
    std::unordered_map<WeightKey, ggml_tensor *, WeightKeyHash> weights;
    std::mutex mu;
};

VkState & state() {
    static VkState s;
    return s;
}

// fused serving hands the wrapper static banks: upload once, read views. Offload slot
// pools change contents under the same pointer, so they keep the per-call gather.
bool g_resident_banks = false;

void op_set_bank_mode(bool resident) { g_resident_banks = resident; }

void ensure_backend() {
    auto & s = state();
    if (s.backend) return;
    s.backend = ggml_backend_vk_init(0);
    if (!s.backend) throw std::runtime_error("ggml vk: ggml_backend_vk_init failed (no Vulkan device?)");
    s.buft = ggml_backend_vk_buffer_type(0);
    ggml_init_params p;
    p.mem_size = 256 * 1024 * 1024;  // weight tensor structs only; data lives in device buffers
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    s.wctx = ggml_init(p);
}

// upload a packed weight once; the ggml tensor wraps the resident device buffer
ggml_tensor * weight_tensor(TensorView w, ggml_type type, int64_t k, int64_t n, int64_t e, int64_t b) {
    auto & s = state();
    WeightKey key;
    key.ptr = reinterpret_cast<uint64_t>(w.data_ptr());
    key.nbytes = w.numel();
    key.type = int(type);
    key.ne[0] = k; key.ne[1] = n; key.ne[2] = e; key.ne[3] = b;
    for (int i = 0; i < 4; i++) key.nb[i] = i < w.ndim() ? w.stride(i) : 0;
    auto it = s.weights.find(key);
    if (it != s.weights.end()) return it->second;

    ggml_tensor * t = ggml_new_tensor_4d(s.wctx, type, k, n, e, b);
    const size_t need = ggml_nbytes(t);
    auto host = to_host(w);
    if (host.size() < need) throw std::runtime_error("ggml vk: weight storage smaller than ggml_nbytes");
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(s.buft, need);
    if (!buf) throw std::runtime_error("ggml vk: weight buffer alloc failed");
    t->buffer = buf;
    t->data = ggml_backend_buffer_get_base(buf);
    ggml_backend_tensor_set(t, host.data(), 0, need);
    s.weights.emplace(key, t);
    return t;
}

// ---------------------------------------------------------------- graph runner

struct GraphRun {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t tmp_gallocr = nullptr;  // large (prefill) graphs are not cached
    std::vector<ggml_backend_buffer_t> input_bufs;
    // uploads are batched into one staging submit: a submit + fence per tensor
    // dominates the decode otherwise
    std::vector<uint8_t> arena;
    std::vector<ggml_tensor *> up_tensors;
    std::vector<size_t> up_offsets, up_sizes;

    ~GraphRun() {
        if (tmp_gallocr) ggml_gallocr_free(tmp_gallocr);
        for (auto b : input_bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }

    void begin(size_t structs_mb = 16) {
        ggml_init_params p;
        p.mem_size = structs_mb * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc = true;
        ctx = ggml_init(p);
        // a prefill routes to many experts: the fused FFN graph exceeds GGML_MAX_NODES
        gf = ggml_new_graph_custom(ctx, 8192, false);
    }

    // an activation tensor with its own device buffer, uploaded from host fp32
    void queue_upload(ggml_tensor * t, const void * data, size_t n) {
        up_offsets.push_back(arena.size());
        arena.insert(arena.end(), static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + n);
        up_tensors.push_back(t);
        up_sizes.push_back(n);
    }

    void flush_uploads() {
        if (up_tensors.empty()) return;
        std::vector<const void *> data(up_tensors.size());
        for (size_t i = 0; i < up_tensors.size(); i++) data[i] = arena.data() + up_offsets[i];
        std::vector<size_t> zeros(up_tensors.size(), 0);
        up_tensors[0]->buffer->iface.set_tensor_n(up_tensors[0]->buffer, up_tensors.data(), data.data(),
                                                  zeros.data(), up_sizes.data(), up_tensors.size());
        up_tensors.clear(); up_offsets.clear(); up_sizes.clear();
    }

    ggml_tensor * input_f32(const float * data, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        const size_t n = ggml_nbytes(t);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(state().buft, n);
        t->buffer = buf;
        t->data = ggml_backend_buffer_get_base(buf);
        queue_upload(t, data, n);
        input_bufs.push_back(buf);
        return t;
    }

    ggml_tensor * input_i32(const int32_t * data, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne0, ne1);
        const size_t n = ggml_nbytes(t);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(state().buft, n);
        t->buffer = buf;
        t->data = ggml_backend_buffer_get_base(buf);
        queue_upload(t, data, n);
        input_bufs.push_back(buf);
        return t;
    }

    ggml_tensor * input_packed(const uint8_t * data, ggml_type type, int64_t k, int64_t n, int64_t e) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, k, n, e);
        const size_t nb = ggml_nbytes(t);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(state().buft, nb);
        t->buffer = buf;
        t->data = ggml_backend_buffer_get_base(buf);
        queue_upload(t, data, nb);
        input_bufs.push_back(buf);
        return t;
    }

    void compute() {
        flush_uploads();
        auto & s = state();
        ggml_gallocr_t alloc;
        if (gf->n_nodes <= 256) {
            auto it = s.gallocrs.find(gf->n_nodes);
            if (it == s.gallocrs.end()) {
                it = s.gallocrs.emplace(gf->n_nodes, ggml_gallocr_new(s.buft)).first;
            }
            alloc = it->second;
        } else {
            tmp_gallocr = ggml_gallocr_new(s.buft);
            alloc = tmp_gallocr;
        }
        if (!ggml_gallocr_alloc_graph(alloc, gf)) throw std::runtime_error("ggml vk: graph alloc failed");
        auto st = ggml_backend_graph_compute(s.backend, gf);
        if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml vk: graph compute failed");
    }

    // download the fp32 result
    std::vector<float> result(ggml_tensor * t) {
        std::vector<float> out(ggml_nelements(t));
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * 4);
        return out;
    }

    // batched download: one staging submit for all results
    std::vector<std::vector<float>> results(const std::vector<ggml_tensor *> & ts) {
        std::vector<std::vector<float>> outs(ts.size());
        std::vector<void *> data(ts.size());
        std::vector<size_t> zeros(ts.size(), 0), sizes(ts.size());
        for (size_t i = 0; i < ts.size(); i++) {
            outs[i].resize(ggml_nelements(ts[i]));
            data[i] = outs[i].data();
            sizes[i] = outs[i].size() * 4;
        }
        ts[0]->buffer->iface.get_tensor_n(ts[0]->buffer, ts.data(), data.data(), zeros.data(), sizes.data(), ts.size());
        return outs;
    }
};

void store_f32(TensorView out, const float * f, int64_t n) {
    if (is_bf16(out.dtype())) {
        std::vector<uint16_t> h(n);
        for (int64_t i = 0; i < n; i++) h[i] = f32_to_bf16(f[i]);
        from_host(out, h.data(), n * 2);
    } else if (is_f16(out.dtype())) {
        std::vector<uint16_t> h(n);
        for (int64_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        from_host(out, h.data(), n * 2);
    } else {
        from_host(out, f, n * 4);
    }
}

// store an [m, n] fp32 result into a strided out view, one row at a time
void store_f32_rows(TensorView out, const float * f, int64_t m, int64_t n) {
    const int64_t esz = dtype_bytes(out.dtype());
    std::vector<uint8_t> row(n * esz);
    for (int64_t mi = 0; mi < m; mi++) {
        for (int64_t ni = 0; ni < n; ni++) {
            const float v = f[mi * n + ni];
            if (is_bf16(out.dtype())) ((uint16_t *) row.data())[ni] = f32_to_bf16(v);
            else if (is_f16(out.dtype())) ((uint16_t *) row.data())[ni] = f32_to_f16(v);
            else ((float *) row.data())[ni] = v;
        }
        void * dst = static_cast<uint8_t *>(out.data_ptr()) + mi * out.stride(0) * esz;
#ifdef FREETOKEN_VK_CUDA
        if (out.device().device_type == kDLCUDA) {
            cudaSetDevice(out.device().device_id);
            cudaMemcpy(dst, row.data(), row.size(), cudaMemcpyHostToDevice);
        } else
#endif
        {
            std::memcpy(dst, row.data(), row.size());
        }
    }
}

// ggml results are [N, mid, T] (index ni + mi*N + ti*N*mid); FreeToken outputs are
// [T, mid, N] row-major
void transpose_last_first(const float * src, float * dst, int64_t n, int64_t mid, int64_t t) {
    for (int64_t ti = 0; ti < t; ti++)
        for (int64_t mi = 0; mi < mid; mi++)
            for (int64_t ni = 0; ni < n; ni++)
                dst[(ti * mid + mi) * n + ni] = src[ni + mi * n + ti * n * mid];
}

// the Vulkan mul_mat_id does not zero -1 routes the way the CUDA kernel does
void zero_inactive(float * out, const std::vector<int32_t> & idh, int64_t t, int64_t top_k, int64_t n) {
    for (int64_t ti = 0; ti < t; ti++)
        for (int64_t j = 0; j < top_k; j++)
            if (idh[ti * top_k + j] < 0)
                std::memset(out + (ti * top_k + j) * n, 0, n * sizeof(float));
}

ggml_glu_op glu_op(const std::string & name) {
    if (name == "swiglu") return GGML_GLU_OP_SWIGLU;
    if (name == "geglu") return GGML_GLU_OP_GEGLU;
    if (name == "swiglu_oai") return GGML_GLU_OP_SWIGLU_OAI;
    if (name == "swiglu_clamp") return GGML_GLU_OP_SWIGLU;  // clamped before the glu
    throw std::runtime_error("ggml vk: unsupported glu " + name);
}

// ---------------------------------------------------------------- ops

void op_mul_mat(TensorView w, int64_t type_id, TensorView x, TensorView out, Optional<TensorView> xq) {
    (void) xq;  // the Vulkan mul_mat quantizes its input inside the pipeline
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
    auto xf = act_to_f32(x);
    GraphRun run;
    run.begin();
    ggml_tensor * wt = weight_tensor(w, ggml_type(type_id), k, n, 1, 1);
    ggml_tensor * xt = run.input_f32(xf.data(), k, m, 1, 1);
    ggml_tensor * res = ggml_mul_mat(run.ctx, wt, xt);  // [n, m]
    ggml_build_forward_expand(run.gf, res);
    run.compute();
    auto r = run.result(res);
    std::vector<float> tr(m * n);
    transpose_last_first(r.data(), tr.data(), n, 1, m);
    // out may be a column-slice view of a packed projection buffer: write row by row
    if (out.stride(0) == n && out.stride(1) == 1) {
        store_f32(out, tr.data(), m * n);
    } else {
        store_f32_rows(out, tr.data(), m, n);
    }
}

// The vendored Vulkan mul_mat_id shader miscomputes leading tokens on this NVIDIA
// driver (small batches lose token 0; larger ones lose the first six), so the route
// GEMM runs as one ggml_mul_mat per routed expert in a single graph instead. Only the
// routed rows cross the host, and nothing is cached: an offload slot pool's contents
// change per call.
struct RouteGroups {
    std::vector<int32_t> experts;                 // unique expert ids, sorted
    std::vector<std::vector<int64_t>> pos;        // flat route positions per expert
    std::vector<std::vector<float>> rows;         // their activations, [Te, k]
};

RouteGroups group_routes(TensorView w, const std::vector<float> & xf, const std::vector<int32_t> & idh,
                         int64_t t, int64_t top_k, int64_t ky, int64_t k) {
    RouteGroups g;
    std::vector<int32_t> uniq;
    for (int32_t id : idh) if (id >= 0) uniq.push_back(id);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    g.experts = uniq;
    g.pos.resize(uniq.size());
    g.rows.resize(uniq.size());
    const int64_t stride0 = w.stride(0);
    const int64_t rowb = w.size(1) * w.size(w.ndim() - 1);
    // gather the routed weight rows once per expert while grouping the activations
    for (size_t i = 0; i < idh.size(); i++) {
        if (idh[i] < 0) continue;
        const size_t e = std::lower_bound(uniq.begin(), uniq.end(), idh[i]) - uniq.begin();
        const int64_t ti = int64_t(i) / top_k, j = int64_t(i) % top_k;
        const int64_t xoff = (ti * ky + (ky == 1 ? 0 : j)) * k;
        g.rows[e].insert(g.rows[e].end(), xf.begin() + xoff, xf.begin() + xoff + k);
        g.pos[e].push_back(int64_t(i));
    }
    (void) stride0; (void) rowb;
    return g;
}

// upload expert e's packed rows [n, row_bytes] as a [k, n] ggml weight
ggml_tensor * expert_weight(GraphRun & run, TensorView w, int64_t type_id, int64_t k, int64_t n, int32_t e) {
    const int64_t rowb = n * w.size(w.ndim() - 1);
    const uint8_t * src = static_cast<const uint8_t *>(w.data_ptr()) + e * w.stride(0);
    std::vector<uint8_t> row(rowb);
    cuda_sync(w.device());
#ifdef FREETOKEN_VK_CUDA
    if (w.device().device_type == kDLCUDA) {
        cudaSetDevice(w.device().device_id);
        cudaMemcpy(row.data(), src, rowb, cudaMemcpyDeviceToHost);
    } else
#endif
    {
        std::memcpy(row.data(), src, rowb);
    }
    return run.input_packed(row.data(), ggml_type(type_id), k, n, 1);
}

// The looped decoder amplifies the Vulkan pipeline's q8_1 activation rounding into
// garbage after a few dozen tokens, so the route GEMMs dequantize the routed rows with
// ggml-quants on the host and run an fp32 GEMM on the device instead. (The in-graph cpy
// dequantize has no K-quant/NVFP4 pipeline on this driver.)
ggml_tensor * expert_gemm_dev(GraphRun & run, TensorView w, int64_t type_id, int64_t k, int64_t n,
                              int32_t e, ggml_tensor * xt) {
    const int64_t te = xt->ne[1];
    if (te <= 8 && ggml_is_quantized(ggml_type(type_id))) {
        // the DMMV vec path (<= mul_mat_vec_max_cols) dot-products fp32 activations
        // against in-shader dequantized weights: exact, and no host dequant round trip
        ggml_tensor * packed;
        if (g_resident_banks) {
            // static bank: one resident device copy, zero-copy per-expert view
            ggml_tensor * bank = weight_tensor(w, ggml_type(type_id), k, n, w.size(0), 1);
            packed = ggml_view_3d(run.ctx, bank, k, n, 1, bank->nb[1], bank->nb[2], e * bank->nb[2]);
        } else {
            packed = expert_weight(run, w, type_id, k, n, e);
        }
        return ggml_mul_mat(run.ctx, packed, xt);
    }
    const int64_t rowb = w.size(w.ndim() - 1);
    const uint8_t * src = static_cast<const uint8_t *>(w.data_ptr()) + e * w.stride(0);
    std::vector<uint8_t> packed(n * rowb);
    cuda_sync(w.device());
#ifdef FREETOKEN_VK_CUDA
    if (w.device().device_type == kDLCUDA) {
        cudaSetDevice(w.device().device_id);
        cudaMemcpy(packed.data(), src, packed.size(), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        std::memcpy(packed.data(), src, packed.size());
    }
    std::vector<float> wf(n * k);
    const auto to_float = ggml_get_type_traits(ggml_type(type_id))->to_float;
    // row dequantization is the hot spot; spread it over the cores
    const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    const int64_t chunk = (n + nth - 1) / nth;
    std::vector<std::future<void>> futs;
    for (unsigned ti = 0; ti < nth; ti++) {
        const int64_t r0 = ti * chunk, r1 = std::min(n, r0 + chunk);
        if (r0 >= r1) break;
        futs.push_back(std::async(std::launch::async, [&, r0, r1] {
            for (int64_t ni = r0; ni < r1; ni++) {
                to_float(packed.data() + ni * rowb, wf.data() + ni * k, k);
            }
        }));
    }
    for (auto & f : futs) f.get();
    ggml_tensor * wt = run.input_f32(wf.data(), k, n, 1, 1);
    return ggml_mul_mat(run.ctx, wt, xt);  // [n, te]
}

ggml_tensor * expert_gemm(GraphRun & run, TensorView w, int64_t type_id, int64_t k, int64_t n,
                          int32_t e, const float * rows, int64_t te) {
    ggml_tensor * xt = run.input_f32(rows, k, te, 1, 1);
    return expert_gemm_dev(run, w, type_id, k, n, e, xt);
}

// gate_up GEMM -> SITU -> down GEMM in ONE graph per call: the looped decoder pays a
// submit + fence per graph, and the split path also round-trips the intermediate
// through the host for the Triton activation. SITU rides ggml unary ops instead:
// (alpha*tanh(g/alpha)) * sigmoid(g) * (limit*tanh(u/limit)), scales folded per expert.
void op_moe_ffn_situ(TensorView wgu, int64_t gu_type, TensorView wdn, int64_t dn_type,
                     TensorView x, TensorView ids, TensorView weights, TensorView out,
                     Optional<TensorView> global_scales, double alpha, double limit) {
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const int64_t t = ids.size(0), top_k = ids.size(1);
    const int64_t k = x.size(1), two_i = wgu.size(1), i = two_i / 2, h = wdn.size(1), e = wgu.size(0);
    auto xf = act_to_f32(x);
    auto idh_t = to_host(ids);
    std::vector<int32_t> idh(t * top_k);
    std::memcpy(idh.data(), idh_t.data(), idh.size() * 4);
    auto wf = act_to_f32(weights);
    std::vector<float> sc;
    if (global_scales.has_value()) {
        auto sb = to_host(global_scales.value());
        sc.resize(sb.size() / 4);
        std::memcpy(sc.data(), sb.data(), sb.size());
    }
    RouteGroups g = group_routes(wgu, xf, idh, t, top_k, 1, k);
    std::vector<float> acc(t * h, 0.0f);
    if (!g.experts.empty()) {
        GraphRun run;
        run.begin(64);
        std::vector<ggml_tensor *> ys;
        for (size_t ei = 0; ei < g.experts.size(); ei++) {
            const int32_t ex = g.experts[ei];
            const int64_t te = int64_t(g.pos[ei].size());
            const float gs = sc.empty() ? 1.0f : sc[ex * 4 + 0];
            const float us = sc.empty() ? 1.0f : sc[ex * 4 + 1];
            const float ds = sc.empty() ? 1.0f : sc[ex * 4 + 2];
            ggml_tensor * xt = run.input_f32(g.rows[ei].data(), k, te, 1, 1);
            ggml_tensor * gu = expert_gemm_dev(run, wgu, gu_type, k, two_i, ex, xt);  // [2i, te]
            // the unary/scale shaders index flat: the interleaved gate/up views must
            // be made contiguous first (a no-op copy for te == 1)
            ggml_tensor * gate = ggml_cont(run.ctx, ggml_view_2d(run.ctx, gu, i, te, gu->nb[1], 0));
            ggml_tensor * up = ggml_cont(run.ctx, ggml_view_2d(run.ctx, gu, i, te, gu->nb[1], i * gu->nb[0]));
            gate = ggml_scale(run.ctx, gate, gs);
            up = ggml_scale(run.ctx, up, us);
            ggml_tensor * tg = ggml_scale(run.ctx, ggml_unary(run.ctx, ggml_scale(run.ctx, gate, float(1.0 / alpha)), GGML_UNARY_OP_TANH), float(alpha));
            ggml_tensor * sg = ggml_unary(run.ctx, gate, GGML_UNARY_OP_SIGMOID);
            ggml_tensor * tu = ggml_scale(run.ctx, ggml_unary(run.ctx, ggml_scale(run.ctx, up, float(1.0 / limit)), GGML_UNARY_OP_TANH), float(limit));
            ggml_tensor * sit = ggml_mul(run.ctx, ggml_mul(run.ctx, tg, sg), tu);  // [i, te]
            ggml_tensor * y = expert_gemm_dev(run, wdn, dn_type, i, h, ex, sit);  // [h, te]
            y = ggml_scale(run.ctx, y, ds);
            ggml_build_forward_expand(run.gf, y);
            ys.push_back(y);
        }
        run.compute();
        auto rs = run.results(ys);
        for (size_t ei = 0; ei < g.experts.size(); ei++) {
            const auto & r = rs[ei];
            const int64_t te = int64_t(g.pos[ei].size());
            for (int64_t ti = 0; ti < te; ti++) {
                const int64_t pos = g.pos[ei][ti];
                const int64_t ti2 = pos / top_k;
                const float w = wf[pos];
                for (int64_t ni = 0; ni < h; ni++) {
                    acc[ti2 * h + ni] += w * r[ni + ti * h];
                }
            }
        }
    }
    store_f32(out, acc.data(), t * h);
}

void op_mul_mat_id(TensorView w, int64_t type_id, TensorView x, TensorView ids, TensorView out,
                   Optional<TensorView> scale, bool inactive) {
    (void) inactive;
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const int64_t t = ids.size(0), top_k = ids.size(1);
    const int64_t k = x.size(2), e = w.size(0), n = w.size(1);
    auto xf = act_to_f32(x);
    auto idh_t = to_host(ids);
    std::vector<int32_t> idh(t * top_k);
    std::memcpy(idh.data(), idh_t.data(), idh.size() * 4);
    std::vector<float> sc;
    if (scale.has_value()) sc = scale_to_host(scale.value(), e);
    RouteGroups g = group_routes(w, xf, idh, t, top_k, x.size(1), k);
    std::vector<float> tr(t * top_k * n, 0.0f);
    if (!g.experts.empty()) {
        GraphRun run;
        run.begin(64);
        std::vector<ggml_tensor *> results;
        for (size_t i = 0; i < g.experts.size(); i++) {
            const int64_t te = int64_t(g.pos[i].size());
            ggml_tensor * res = expert_gemm(run, w, type_id, k, n, g.experts[i], g.rows[i].data(), te);
            ggml_build_forward_expand(run.gf, res);
            results.push_back(res);
        }
        run.compute();
        auto rs = run.results(results);
        for (size_t i = 0; i < g.experts.size(); i++) {
            const auto & r = rs[i];
            const float s = sc.empty() ? 1.0f : sc[g.experts[i]];
            const int64_t te = int64_t(g.pos[i].size());
            for (int64_t ti = 0; ti < te; ti++)
                for (int64_t ni = 0; ni < n; ni++)
                    tr[g.pos[i][ti] * n + ni] = r[ni + ti * n] * s;
        }
    }
    store_f32(out, tr.data(), t * top_k * n);
}

void op_mul_mat_id_glu(TensorView w, int64_t type_id, TensorView x, TensorView ids, TensorView out, String glu,
                       double limit, Optional<TensorView> gate_scale, Optional<TensorView> up_scale) {
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const std::string gname(glu);
    const int64_t t = ids.size(0), top_k = ids.size(1);
    const int64_t k = x.size(1), e = w.size(0), n = w.size(1) / 2;
    auto xf = act_to_f32(x);
    auto idh_t = to_host(ids);
    std::vector<int32_t> idh(t * top_k);
    std::memcpy(idh.data(), idh_t.data(), idh.size() * 4);
    std::vector<float> gsc, usc;
    if (gate_scale.has_value()) gsc = scale_to_host(gate_scale.value(), e);
    if (up_scale.has_value()) usc = scale_to_host(up_scale.value(), e);
    RouteGroups g = group_routes(w, xf, idh, t, top_k, 1, k);
    std::vector<float> tr(t * top_k * n, 0.0f);
    if (!g.experts.empty()) {
        GraphRun run;
        run.begin(64);
        std::vector<ggml_tensor *> results;
        for (size_t i = 0; i < g.experts.size(); i++) {
            const int64_t te = int64_t(g.pos[i].size());
            ggml_tensor * res = expert_gemm(run, w, type_id, k, 2 * n, g.experts[i], g.rows[i].data(), te);
            ggml_build_forward_expand(run.gf, res);
            results.push_back(res);
        }
        run.compute();
        auto rs = run.results(results);
        // gated activation on the host: the route outputs are small, and the fused glu
        // epilogue rides the same mul_mat_id shader this path avoids
        for (size_t i = 0; i < g.experts.size(); i++) {
            const auto & r = rs[i];
            const int32_t ex = g.experts[i];
            const float gs = gsc.empty() ? 1.0f : gsc[ex];
            const float us = usc.empty() ? 1.0f : usc[ex];
            const int64_t te = int64_t(g.pos[i].size());
            for (int64_t ti = 0; ti < te; ti++) {
                float * dst = tr.data() + g.pos[i][ti] * n;
                for (int64_t ni = 0; ni < n; ni++) {
                    float gate = r[ni + ti * 2 * n] * gs;
                    float up = r[n + ni + ti * 2 * n] * us;
                    if (gname == "swiglu_clamp") {
                        gate = std::max(float(-limit), std::min(float(limit), gate));
                        up = std::max(float(-limit), std::min(float(limit), up));
                    }
                    float a;
                    if (gname == "geglu") {
                        a = 0.5f * gate * (1.0f + std::tanh(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
                    } else if (gname == "swiglu_oai") {
                        a = gate / (1.0f + std::exp(-1.702f * gate));
                    } else {  // swiglu (and the clamped variant)
                        a = gate / (1.0f + std::exp(-gate));
                    }
                    dst[ni] = a * up;
                }
            }
        }
    }
    store_f32(out, tr.data(), t * top_k * n);
}

void op_moe_weighted_sum(TensorView y, TensorView weights, TensorView out) {
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const int64_t t = y.size(0), top_k = y.size(1), n = y.size(2);
    auto yf = act_to_f32(y);
    auto wf = act_to_f32(weights);
    GraphRun run;
    run.begin();
    ggml_tensor * yt = run.input_f32(yf.data(), n, top_k, t, 1);
    ggml_tensor * wt = run.input_f32(wf.data(), 1, top_k, t, 1);
    ggml_tensor * prod = ggml_mul(run.ctx, yt, wt);          // [n, k, t]
    // ggml_sum_rows reduces ne[0], so put the route dim first and make it contiguous
    ggml_tensor * trt = ggml_cont(run.ctx, ggml_transpose(run.ctx, prod));  // [k, n, t]
    ggml_tensor * res = ggml_sum_rows(run.ctx, trt);         // [1, n, t]
    ggml_build_forward_expand(run.gf, res);
    run.compute();
    auto r = run.result(res);
    std::vector<float> tr(t * n);
    transpose_last_first(r.data(), tr.data(), n, 1, t);
    store_f32(out, tr.data(), t * n);
}

void op_dequantize(TensorView w, int64_t type_id, int64_t k, TensorView out) {
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const int64_t rows = w.numel() / w.size(w.ndim() - 1);
    auto host = to_host(w);
    const int64_t rowb = w.size(w.ndim() - 1);
    // chunked so the transient device buffers stay small next to a loaded model
    const int64_t chunk = 16384;
    std::vector<float> res(rows * k);
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, rows - r0);
        GraphRun run;
        run.begin();
        ggml_tensor * wt = run.input_packed(host.data() + r0 * rowb, ggml_type(type_id), k, nr, 1);
        ggml_tensor * dst = ggml_new_tensor_2d(run.ctx, GGML_TYPE_F32, k, nr);
        ggml_tensor * cpy = ggml_cpy(run.ctx, wt, dst);
        ggml_build_forward_expand(run.gf, cpy);
        run.compute();
        auto r = run.result(cpy);
        std::copy(r.begin(), r.end(), res.begin() + r0 * k);
    }
    store_f32(out, res.data(), rows * k);
}

void op_get_rows(TensorView w, int64_t type_id, int64_t k, TensorView ids, TensorView out) {
    std::lock_guard<std::mutex> lock(state().mu);
    ensure_backend();
    const int64_t rows = w.numel() / w.size(w.ndim() - 1);
    const int64_t m = ids.size(0);
    auto idh_t = to_host(ids);
    std::vector<int32_t> idh(m);
    std::memcpy(idh.data(), idh_t.data(), m * 4);
    GraphRun run;
    run.begin();
    ggml_tensor * wt = weight_tensor(w, ggml_type(type_id), k, rows, 1, 1);
    ggml_tensor * idt = run.input_i32(idh.data(), m, 1);
    ggml_tensor * res = ggml_get_rows(run.ctx, wt, idt);  // [k, m]
    ggml_build_forward_expand(run.gf, res);
    run.compute();
    auto r = run.result(res);
    std::vector<float> tr(m * k);
    transpose_last_first(r.data(), tr.data(), k, 1, m);
    store_f32(out, tr.data(), m * k);
}

void op_quantize_q8_1(TensorView x, TensorView out) {
    // the Vulkan mul_mat quantizes internally; keep the fp32 rows so the shared-xq
    // contract holds without a q8_1 layout the Vulkan pipeline does not read
    auto xf = act_to_f32(x);
    from_host(out, xf.data(), xf.size() * 4);
}

int64_t q8_1_row_bytes(int64_t k) { return k * 4; }
int64_t mmq_tail_bytes(int64_t type_id, int64_t k) { (void) type_id; (void) k; return 0; }
bool op_uses_mmvq(int64_t type_id, int64_t m) { (void) type_id; (void) m; return true; }
int64_t op_mmvq_max_tokens(int64_t type_id) { (void) type_id; return 1 << 20; }

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_q8_1, op_quantize_q8_1);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8_1_row_bytes, q8_1_row_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mmq_tail_bytes, mmq_tail_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat, op_mul_mat);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat_id, op_mul_mat_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat_id_glu, op_mul_mat_id_glu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_weighted_sum, op_moe_weighted_sum);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequantize, op_dequantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_rows, op_get_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(uses_mmvq, op_uses_mmvq);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mmvq_max_tokens, op_mmvq_max_tokens);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_bank_mode, op_set_bank_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_ffn_situ, op_moe_ffn_situ);
