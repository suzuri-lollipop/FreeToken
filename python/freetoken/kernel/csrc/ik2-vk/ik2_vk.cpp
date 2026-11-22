// ik2-vk: High-throughput Vulkan 1.2 compute ops for FreeToken (BDA + subgroupAdd +
// zero-copy CUDA<->Vulkan external memory & external semaphores).
//
// Zero descriptor sets: every buffer is a 64-bit GPU address in push constants.
// Zero per-call Vulkan allocations: persistent external-memory IO arena shared with
// CUDA, persistent device scratch, 4-slot command buffer / fence ring.
// Supports NVFP4 (40), Q8_0 (8), BF16 (30), F32 (0), Q4_0 (2), Q6_K (14).

#include <vulkan/vulkan.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ik2_shaders.h"

#ifdef FREETOKEN_VK_CUDA
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <unistd.h>
#endif

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace {

#define VK_CHECK(x) do { \
    VkResult r_ = (x); \
    if (r_ != VK_SUCCESS) throw std::runtime_error(std::string("ik2_vk: " #x " failed (") + std::to_string(r_) + ")"); \
} while (0)

#ifdef FREETOKEN_VK_CUDA
#define CU_CHECK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) throw std::runtime_error(std::string("ik2_vk CUDA: ") + cudaGetErrorString(e_)); \
} while (0)
#endif

// Must match push_constant layouts in shaders/*.comp (all <= 128 bytes)
struct GemvPC {
    uint64_t w_addr, x_addr, y_addr;
    int32_t K, N, M, ty, x_dt, y_dt, y_stride;
};

struct MoeGuSituPC {
    uint64_t w_addr, x_addr, ids_addr, sc_addr, inter_addr;
    int32_t K, I, top_k, T, ty, x_dt;
    float gcap, ucap;
};

struct MoeDnPC {
    uint64_t w_addr, x_addr, ids_addr, sc_addr, y_addr;
    int32_t K, H, top_k, T, ty, x_dt, y_dt, x_per_route, sc_stride, sc_col;
};

struct WsumPC {
    uint64_t y_addr, ids_addr, wt_addr, out_addr;
    int32_t H, top_k, T, y_dt, out_dt;
};

struct GetRowsPC {
    uint64_t w_addr, ids_addr, out_addr;
    int32_t K, M, ty, out_dt;
};

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    uint64_t bda = 0;
    void * mapped = nullptr;
    size_t size = 0;
#ifdef FREETOKEN_VK_CUDA
    cudaExternalMemory_t cu_ext = nullptr;
    void * cu_ptr = nullptr;
#endif
};

static constexpr int kRing = 4;

struct State {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd[kRing] = {};
    VkFence fence[kRing] = {};
    bool fence_pending[kRing] = {};
    int ring_idx = 0;

    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline p_gemv = VK_NULL_HANDLE;
    VkPipeline p_gu_situ = VK_NULL_HANDLE;
    VkPipeline p_moe_dn = VK_NULL_HANDLE;
    VkPipeline p_wsum = VK_NULL_HANDLE;
    VkPipeline p_get_rows = VK_NULL_HANDLE;

    Buf io;          // shared external VRAM (CUDA<->Vulkan zero-copy) or device-local
    Buf host_io;     // host-visible staging (for CPU tensors and bank uploads)
    Buf scratch;     // device-local scratch for MoE inter & y
    std::unordered_map<uint64_t, Buf> banks;

    bool has_cuda_interop = false;
#ifdef FREETOKEN_VK_CUDA
    int cu_device = 0;
    VkSemaphore v_c2v = VK_NULL_HANDLE;
    VkSemaphore v_v2c = VK_NULL_HANDLE;
    cudaExternalSemaphore_t c_c2v = nullptr;
    cudaExternalSemaphore_t c_v2c = nullptr;
#endif

    bool resident_banks = false;
    std::mutex mu;
    bool ready = false;
};

State & st() {
    static State s;
    return s;
}

int dl_to_dt(DLDataType dt) {
    if (dt.code == kDLFloat && dt.bits == 32) return 0;
    if (dt.code == kDLBfloat && dt.bits == 16) return 1;
    if (dt.code == kDLFloat && dt.bits == 16) return 2;
    throw std::runtime_error("ik2_vk: unsupported float dtype");
}

size_t dt_bytes(DLDataType dt) { return size_t(dt.bits / 8); }

size_t align_up(size_t v, size_t a = 256) { return (v + a - 1) & ~(a - 1); }

size_t ggml_row_bytes_for(int ty, int K) {
    switch (ty) {
        case 40: return size_t(K / 64) * 36;
        case 8:  return size_t(K / 32) * 34;
        case 30: return size_t(K) * 2;
        case 0:  return size_t(K) * 4;
        case 2:  return size_t(K / 32) * 18;
        case 14: return size_t(K / 256) * 210;
        default: throw std::runtime_error("ik2_vk: unsupported ggml type " + std::to_string(ty));
    }
}

uint32_t mem_type(uint32_t bits, VkFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(st().phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("ik2_vk: no suitable memory type");
}

Buf make_buf(size_t size, bool host, bool export_cuda = false) {
    auto & s = st();
    size = std::max<size_t>(size, 256);
    Buf bf;
    bf.size = size;

    VkExternalMemoryBufferCreateInfo embi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    embi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    if (export_cuda) bi.pNext = &embi;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(s.dev, &bi, nullptr, &bf.b));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(s.dev, bf.b, &mr);

    VkExportMemoryAllocateInfo emai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateFlagsInfo mafi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    mafi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    if (export_cuda) mafi.pNext = &emai;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &mafi;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mem_type(mr.memoryTypeBits,
        host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(s.dev, &ai, nullptr, &bf.m));
    VK_CHECK(vkBindBufferMemory(s.dev, bf.b, bf.m, 0));

    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = bf.b;
    bf.bda = vkGetBufferDeviceAddress(s.dev, &bdai);

    if (host) VK_CHECK(vkMapMemory(s.dev, bf.m, 0, size, 0, &bf.mapped));

#ifdef FREETOKEN_VK_CUDA
    if (export_cuda) {
        auto pGetMemFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(s.dev, "vkGetMemoryFdKHR"));
        VkMemoryGetFdInfoKHR mfi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        mfi.memory = bf.m;
        mfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        VK_CHECK(pGetMemFd(s.dev, &mfi, &fd));
        CU_CHECK(cudaSetDevice(s.cu_device));
        cudaExternalMemoryHandleDesc desc{};
        desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = fd;
        desc.size = mr.size;
        CU_CHECK(cudaImportExternalMemory(&bf.cu_ext, &desc));
        cudaExternalMemoryBufferDesc mbd{};
        mbd.offset = 0;
        mbd.size = size;
        CU_CHECK(cudaExternalMemoryGetMappedBuffer(&bf.cu_ptr, bf.cu_ext, &mbd));
    }
#endif
    return bf;
}

void wait_all_idle() {
    auto & s = st();
    VK_CHECK(vkQueueWaitIdle(s.queue));
    for (int i = 0; i < kRing; i++) {
        if (s.fence_pending[i]) {
            VK_CHECK(vkWaitForFences(s.dev, 1, &s.fence[i], VK_TRUE, UINT64_MAX));
            VK_CHECK(vkResetFences(s.dev, 1, &s.fence[i]));
            s.fence_pending[i] = false;
        }
    }
}

void free_buf(Buf & bf) {
    auto & s = st();
#ifdef FREETOKEN_VK_CUDA
    if (bf.cu_ptr) { cudaFree(bf.cu_ptr); bf.cu_ptr = nullptr; }
    if (bf.cu_ext) { cudaDestroyExternalMemory(bf.cu_ext); bf.cu_ext = nullptr; }
#endif
    if (bf.b) vkDestroyBuffer(s.dev, bf.b, nullptr);
    if (bf.m) vkFreeMemory(s.dev, bf.m, nullptr);
    bf = Buf{};
}

void ensure_buf(Buf & bf, size_t need, bool host, bool export_cuda = false) {
    if (bf.size >= need) return;
    wait_all_idle();
#ifdef FREETOKEN_VK_CUDA
    if (export_cuda && st().has_cuda_interop) {
        cudaSetDevice(st().cu_device);
        cudaDeviceSynchronize();
    }
#endif
    size_t next = std::max(need, std::max<size_t>(bf.size * 2, 16 * 1024 * 1024));
    free_buf(bf);
    bf = make_buf(next, host, export_cuda);
}

VkPipeline make_pipeline(const uint32_t * code, size_t bytes) {
    auto & s = st();
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = bytes;
    sci.pCode = code;
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(s.dev, &sci, nullptr, &mod));
    VkPipelineShaderStageCreateInfo stg{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stg.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stg.module = mod;
    stg.pName = "main";
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pci.stage = stg;
    pci.layout = s.pl;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(s.dev, VK_NULL_HANDLE, 1, &pci, nullptr, &p));
    vkDestroyShaderModule(s.dev, mod, nullptr);
    return p;
}

void init(int device_override = -1) {
    auto & s = st();
    if (s.ready) return;

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "freetoken-ik2-vk";
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &s.inst));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(s.inst, &n, nullptr);
    if (n == 0) throw std::runtime_error("ik2_vk: no Vulkan device found");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(s.inst, &n, devs.data());

    // Match Vulkan physical device to active CUDA device by UUID when CUDA is present
#ifdef FREETOKEN_VK_CUDA
    int cu_cnt = 0;
    if (device_override < 0 && cudaGetDeviceCount(&cu_cnt) == cudaSuccess && cu_cnt > 0) {
        if (cudaGetDevice(&s.cu_device) == cudaSuccess) {
            cudaDeviceProp cprop;
            if (cudaGetDeviceProperties(&cprop, s.cu_device) == cudaSuccess) {
                for (auto d : devs) {
                    VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                    p2.pNext = &idp;
                    vkGetPhysicalDeviceProperties2(d, &p2);
                    if (std::memcmp(idp.deviceUUID, cprop.uuid.bytes, 16) == 0) {
                        s.phys = d;
                        s.has_cuda_interop = true;
                        break;
                    }
                }
            }
        }
    }
#endif
    if (s.phys == VK_NULL_HANDLE) {
        int idx = device_override >= 0 ? device_override : 0;
        if (idx >= int(n)) throw std::runtime_error("ik2_vk: invalid Vulkan device index");
        s.phys = devs[idx];
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &qn, qp.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; i++) {
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { s.qfam = i; found = true; break; }
    }
    if (!found) throw std::runtime_error("ik2_vk: no compute queue");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = s.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char *> exts;
    if (s.has_cuda_interop) {
        exts = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        };
    }

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.pNext = &f12;
    f11.storageBuffer16BitAccess = VK_TRUE;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(exts.size());
    dci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
    VK_CHECK(vkCreateDevice(s.phys, &dci, nullptr, &s.dev));
    vkGetDeviceQueue(s.dev, s.qfam, 0, &s.queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = s.qfam;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(s.dev, &pci, nullptr, &s.pool));

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = s.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = kRing;
    VK_CHECK(vkAllocateCommandBuffers(s.dev, &cbi, s.cmd));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (int i = 0; i < kRing; i++) VK_CHECK(vkCreateFence(s.dev, &fci, nullptr, &s.fence[i]));

    // Empty descriptor set layout + 128-byte push constants (BDA pointers + dims)
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(s.dev, &plci, nullptr, &s.pl));

    s.p_gemv = make_pipeline(ik2_gemv_spv, ik2_gemv_spv_len);
    s.p_gu_situ = make_pipeline(ik2_moe_gu_situ_spv, ik2_moe_gu_situ_spv_len);
    s.p_moe_dn = make_pipeline(ik2_moe_dn_spv, ik2_moe_dn_spv_len);
    s.p_wsum = make_pipeline(ik2_wsum_spv, ik2_wsum_spv_len);
    s.p_get_rows = make_pipeline(ik2_get_rows_spv, ik2_get_rows_spv_len);

#ifdef FREETOKEN_VK_CUDA
    if (s.has_cuda_interop) {
        auto pGetSemFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(s.dev, "vkGetSemaphoreFdKHR"));
        auto make_sem = [&](VkSemaphore & vs, cudaExternalSemaphore_t & cs) {
            VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
            esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            sci.pNext = &esci;
            VK_CHECK(vkCreateSemaphore(s.dev, &sci, nullptr, &vs));
            VkSemaphoreGetFdInfoKHR sfi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            sfi.semaphore = vs;
            sfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            int fd = -1;
            VK_CHECK(pGetSemFd(s.dev, &sfi, &fd));
            CU_CHECK(cudaSetDevice(s.cu_device));
            cudaExternalSemaphoreHandleDesc desc{};
            desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
            desc.handle.fd = fd;
            CU_CHECK(cudaImportExternalSemaphore(&cs, &desc));
        };
        make_sem(s.v_c2v, s.c_c2v);
        make_sem(s.v_v2c, s.c_v2c);
        s.io = make_buf(64 * 1024 * 1024, false, true);
    }
#endif
    s.host_io = make_buf(32 * 1024 * 1024, true, false);
    s.scratch = make_buf(32 * 1024 * 1024, false, false);
    s.ready = true;
}

#ifdef FREETOKEN_VK_CUDA
cudaStream_t cu_stream(DLDevice d) {
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(d.device_type, d.device_id));
    return stream == nullptr ? cudaStreamLegacy : stream;
}
#endif

// Contiguous host copy of a possibly strided tensor (used once at weight upload time)
std::vector<uint8_t> gather_to_host(TensorView t) {
    const int64_t esz = int64_t(dt_bytes(t.dtype()));
    const int64_t n = t.numel() * esz;
    std::vector<uint8_t> h(n);
    if (n == 0) return h;
    const int nd = t.ndim();
    std::vector<int64_t> shape(nd), stride(nd);
    int64_t span = esz;
    bool contig = true;
    int64_t exp_s = esz;
    for (int i = nd - 1; i >= 0; i--) {
        shape[i] = t.size(i);
        stride[i] = t.stride(i) * esz;
        if (shape[i] > 1 && stride[i] != exp_s) contig = false;
        exp_s *= shape[i];
        span += (shape[i] - 1) * stride[i];
    }
    if (contig) {
#ifdef FREETOKEN_VK_CUDA
        if (t.device().device_type == kDLCUDA) {
            CU_CHECK(cudaSetDevice(t.device().device_id));
            CU_CHECK(cudaStreamSynchronize(cu_stream(t.device())));
            CU_CHECK(cudaMemcpy(h.data(), t.data_ptr(), n, cudaMemcpyDeviceToHost));
            return h;
        }
#endif
        std::memcpy(h.data(), t.data_ptr(), n);
        return h;
    }
    std::vector<uint8_t> raw(span);
#ifdef FREETOKEN_VK_CUDA
    if (t.device().device_type == kDLCUDA) {
        CU_CHECK(cudaSetDevice(t.device().device_id));
        CU_CHECK(cudaStreamSynchronize(cu_stream(t.device())));
        CU_CHECK(cudaMemcpy(raw.data(), t.data_ptr(), span, cudaMemcpyDeviceToHost));
    } else
#endif
    {
        std::memcpy(raw.data(), t.data_ptr(), span);
    }
    const int64_t run = shape[nd - 1] * esz;
    int64_t pos = 0;
    std::vector<int64_t> idx(nd - 1, 0);
    while (pos < n) {
        int64_t off = 0;
        for (int i = 0; i + 1 < nd; i++) off += idx[i] * stride[i];
        std::memcpy(h.data() + pos, raw.data() + off, run);
        pos += run;
        int d = nd - 2;
        while (d >= 0 && ++idx[d] >= shape[d]) { idx[d] = 0; d--; }
        if (d < 0) break;
    }
    return h;
}

uint64_t sample_hash(const uint8_t * p, size_t bytes) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(bytes);
    const size_t n = std::min<size_t>(bytes, 64);
    for (size_t i = 0; i + 8 <= n; i += 8) {
        uint64_t v;
        std::memcpy(&v, p + i, 8);
        mix(v);
    }
    return h;
}

// Upload a weight buffer once into Vulkan device-local memory and return its BDA.
uint64_t cached_weight_bda(TensorView w) {
    auto & s = st();
    const size_t bytes = size_t(w.numel()) * dt_bytes(w.dtype());
    uint64_t key = reinterpret_cast<uint64_t>(w.data_ptr());
    key ^= uint64_t(bytes) * 1099511628211ull;
    for (int i = 0; i < w.ndim(); i++) key ^= uint64_t(w.size(i)) * (31u + i) ^ (uint64_t(w.stride(i)) * (131u + i));
    if (w.device().device_type == kDLCPU) {
        const uint8_t * base = static_cast<const uint8_t *>(w.data_ptr());
        key ^= sample_hash(base, bytes);
        key ^= sample_hash(base + bytes / 2, bytes - bytes / 2) * 31;
        key ^= sample_hash(base + (bytes > 64 ? bytes - 64 : 0), bytes > 64 ? 64 : bytes) * 131;
    }
    auto it = s.banks.find(key);
    if (it != s.banks.end()) return it->second.bda;

    // Upload via chunked host staging so we never allocate a giant staging buffer
    Buf bf = make_buf(bytes, false, false);
    auto host = gather_to_host(w);
    const size_t chunk = std::min<size_t>(bytes, 64 * 1024 * 1024);
    ensure_buf(s.host_io, chunk, true, false);
    wait_all_idle();

    VkCommandBuffer cb = s.cmd[0];
    for (size_t off = 0; off < bytes; off += chunk) {
        size_t sz = std::min(chunk, bytes - off);
        std::memcpy(s.host_io.mapped, host.data() + off, sz);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        VkBufferCopy region{0, off, sz};
        vkCmdCopyBuffer(cb, s.host_io.b, bf.b, 1, &region);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(s.queue, 1, &si, s.fence[0]));
        VK_CHECK(vkWaitForFences(s.dev, 1, &s.fence[0], VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(s.dev, 1, &s.fence[0]));
    }
    s.banks.emplace(key, bf);
    return bf.bda;
}

// Acquire next command buffer slot in the ring (waits only if that slot is still in flight)
VkCommandBuffer acquire_cmd(int & slot_out) {
    auto & s = st();
    int slot = s.ring_idx;
    s.ring_idx = (s.ring_idx + 1) % kRing;
    if (s.fence_pending[slot]) {
        VK_CHECK(vkWaitForFences(s.dev, 1, &s.fence[slot], VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(s.dev, 1, &s.fence[slot]));
        s.fence_pending[slot] = false;
    }
    slot_out = slot;
    VkCommandBuffer cb = s.cmd[slot];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

void cmd_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

template <typename T>
void cmd_dispatch(VkCommandBuffer cb, VkPipeline pipe, const T & pc, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cb, st().pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(T), &pc);
    vkCmdDispatch(cb, gx, gy, gz);
}

// Submit command buffer: async external-semaphore handoff for CUDA, fence wait for CPU.
void submit_cmd(int slot, bool use_cuda, DLDevice dev) {
    auto & s = st();
    VK_CHECK(vkEndCommandBuffer(s.cmd[slot]));
#ifdef FREETOKEN_VK_CUDA
    if (use_cuda && s.has_cuda_interop) {
        cudaStream_t stream = cu_stream(dev);
        cudaExternalSemaphoreSignalParams sp{};
        CU_CHECK(cudaSignalExternalSemaphoresAsync(&s.c_c2v, &sp, 1, stream));

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &s.v_c2v;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s.cmd[slot];
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &s.v_v2c;
        VK_CHECK(vkQueueSubmit(s.queue, 1, &si, s.fence[slot]));
        s.fence_pending[slot] = true;

        cudaExternalSemaphoreWaitParams wp{};
        CU_CHECK(cudaWaitExternalSemaphoresAsync(&s.c_v2c, &wp, 1, stream));
        return;
    }
#endif
    (void) use_cuda;
    (void) dev;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd[slot];
    VK_CHECK(vkQueueSubmit(s.queue, 1, &si, s.fence[slot]));
    VK_CHECK(vkWaitForFences(s.dev, 1, &s.fence[slot], VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(s.dev, 1, &s.fence[slot]));
}

// Helper to stage a CUDA or CPU tensor into the active IO arena (`s.io` for CUDA, `s.host_io` for CPU)
struct IoArena {
    Buf * buf = nullptr;
    size_t cursor = 0;
    bool use_cuda = false;
    DLDevice dev{kDLCPU, 0};

    void begin(size_t total_bytes, bool cuda, DLDevice d) {
        auto & s = st();
        use_cuda = cuda && s.has_cuda_interop;
        dev = d;
        if (use_cuda) {
            ensure_buf(s.io, total_bytes, false, true);
            buf = &s.io;
        } else {
            ensure_buf(s.host_io, total_bytes, true, false);
            buf = &s.host_io;
        }
        cursor = 0;
    }

    size_t reserve(size_t bytes) {
        size_t off = cursor;
        cursor = align_up(cursor + bytes);
        return off;
    }

    uint64_t bda(size_t off) const { return buf->bda + off; }

    // Stage contiguous tensor `t` into arena at `off`
    void put(size_t off, TensorView t) {
        size_t n = size_t(t.numel()) * dt_bytes(t.dtype());
        if (n == 0) return;
#ifdef FREETOKEN_VK_CUDA
        if (use_cuda) {
            cudaStream_t stream = cu_stream(dev);
            cudaMemcpyKind kind = (t.device().device_type == kDLCUDA) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            CU_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(buf->cu_ptr) + off, t.data_ptr(), n, kind, stream));
            return;
        }
        if (t.device().device_type == kDLCUDA) {
            CU_CHECK(cudaSetDevice(t.device().device_id));
            CU_CHECK(cudaStreamSynchronize(cu_stream(t.device())));
            CU_CHECK(cudaMemcpy(static_cast<uint8_t *>(buf->mapped) + off, t.data_ptr(), n, cudaMemcpyDeviceToHost));
            return;
        }
#endif
        std::memcpy(static_cast<uint8_t *>(buf->mapped) + off, t.data_ptr(), n);
    }

    // Copy 2D output `[rows, cols]` from arena `off` to possibly row-strided `out`
    void get_2d(size_t off, TensorView out, int64_t rows, int64_t cols) {
        size_t esz = dt_bytes(out.dtype());
        size_t row_bytes = size_t(cols) * esz;
        size_t dst_pitch = size_t(out.ndim() >= 2 ? out.stride(0) : cols) * esz;
        if (rows == 0 || cols == 0) return;
#ifdef FREETOKEN_VK_CUDA
        if (use_cuda) {
            cudaStream_t stream = cu_stream(dev);
            if (dst_pitch == row_bytes || rows == 1) {
                CU_CHECK(cudaMemcpyAsync(out.data_ptr(), static_cast<const uint8_t *>(buf->cu_ptr) + off,
                                         size_t(rows) * row_bytes, cudaMemcpyDeviceToDevice, stream));
            } else {
                CU_CHECK(cudaMemcpy2DAsync(out.data_ptr(), dst_pitch,
                                           static_cast<const uint8_t *>(buf->cu_ptr) + off, row_bytes,
                                           row_bytes, size_t(rows), cudaMemcpyDeviceToDevice, stream));
            }
            return;
        }
        if (out.device().device_type == kDLCUDA) {
            CU_CHECK(cudaSetDevice(out.device().device_id));
            if (dst_pitch == row_bytes || rows == 1) {
                CU_CHECK(cudaMemcpy(out.data_ptr(), static_cast<const uint8_t *>(buf->mapped) + off,
                                    size_t(rows) * row_bytes, cudaMemcpyHostToDevice));
            } else {
                CU_CHECK(cudaMemcpy2D(out.data_ptr(), dst_pitch,
                                      static_cast<const uint8_t *>(buf->mapped) + off, row_bytes,
                                      row_bytes, size_t(rows), cudaMemcpyHostToDevice));
            }
            return;
        }
#endif
        const uint8_t * src = static_cast<const uint8_t *>(buf->mapped) + off;
        uint8_t * dst = static_cast<uint8_t *>(out.data_ptr());
        for (int64_t r = 0; r < rows; r++) std::memcpy(dst + r * dst_pitch, src + r * row_bytes, row_bytes);
    }

    void get_flat(size_t off, TensorView out) {
        size_t n = size_t(out.numel()) * dt_bytes(out.dtype());
        if (n == 0) return;
#ifdef FREETOKEN_VK_CUDA
        if (use_cuda) {
            CU_CHECK(cudaMemcpyAsync(out.data_ptr(), static_cast<const uint8_t *>(buf->cu_ptr) + off,
                                     n, cudaMemcpyDeviceToDevice, cu_stream(dev)));
            return;
        }
        if (out.device().device_type == kDLCUDA) {
            CU_CHECK(cudaSetDevice(out.device().device_id));
            CU_CHECK(cudaMemcpy(out.data_ptr(), static_cast<const uint8_t *>(buf->mapped) + off,
                                n, cudaMemcpyHostToDevice));
            return;
        }
#endif
        std::memcpy(out.data_ptr(), static_cast<const uint8_t *>(buf->mapped) + off, n);
    }
};

// ---------------------------------------------------------------- exported ops

void op_init(int64_t device) {
    std::lock_guard<std::mutex> lock(st().mu);
    init(int(device));
}

void op_set_bank_mode(bool resident) {
    std::lock_guard<std::mutex> lock(st().mu);
    st().resident_banks = resident;
}

void op_mul_mat(TensorView w, int64_t ty, TensorView x, TensorView out) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    const bool use_cuda = (x.device().device_type == kDLCUDA);
    const size_t w_bytes = size_t(w.numel()) * dt_bytes(w.dtype());
    // Tiny CUDA weights (e.g. unit tests whose allocator reuses pointers) stage D2D
    // directly instead of caching; real model weights (>= 256 KiB) stay cached in VRAM.
    const bool stage_w = (w.device().device_type == kDLCUDA && w_bytes <= 256 * 1024 && !s.resident_banks);
    const uint64_t w_bda_cached = stage_w ? 0ull : cached_weight_bda(w);

    const size_t x_bytes = size_t(M * K) * dt_bytes(x.dtype());
    const size_t y_bytes = size_t(M * N) * dt_bytes(out.dtype());
    IoArena io;
    io.begin((stage_w ? align_up(w_bytes) : 0) + align_up(x_bytes) + align_up(y_bytes), use_cuda, x.device());
    size_t w_off = stage_w ? io.reserve(w_bytes) : 0;
    size_t x_off = io.reserve(x_bytes);
    size_t y_off = io.reserve(y_bytes);
    if (stage_w) {
        auto wh = gather_to_host(w);
#ifdef FREETOKEN_VK_CUDA
        if (io.use_cuda) {
            CU_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(io.buf->cu_ptr) + w_off, wh.data(),
                                     w_bytes, cudaMemcpyHostToDevice, cu_stream(x.device())));
        } else
#endif
        {
            std::memcpy(static_cast<uint8_t *>(io.buf->mapped) + w_off, wh.data(), w_bytes);
        }
    }
    io.put(x_off, x);

    int slot = 0;
    VkCommandBuffer cb = acquire_cmd(slot);
    GemvPC pc{stage_w ? io.bda(w_off) : w_bda_cached, io.bda(x_off), io.bda(y_off),
              int32_t(K), int32_t(N), int32_t(M),
              int32_t(ty), dl_to_dt(x.dtype()), dl_to_dt(out.dtype()), int32_t(N)};
    cmd_dispatch(cb, s.p_gemv, pc, uint32_t((N + 3) / 4), uint32_t(M));
    submit_cmd(slot, use_cuda, x.device());
    io.get_2d(y_off, out, M, N);
}

// Batched dense projections (up to 6) sharing the same input `x` in ONE command buffer & ONE handoff!
void op_mul_mat_batch(
    TensorView x,
    TensorView w0, int64_t t0, TensorView o0,
    Optional<TensorView> w1, int64_t t1, Optional<TensorView> o1,
    Optional<TensorView> w2, int64_t t2, Optional<TensorView> o2,
    Optional<TensorView> w3, int64_t t3, Optional<TensorView> o3,
    Optional<TensorView> w4, int64_t t4, Optional<TensorView> o4,
    Optional<TensorView> w5, int64_t t5, Optional<TensorView> o5
) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();
    struct Item { TensorView w; int64_t ty; TensorView o; };
    std::vector<Item> items;
    items.reserve(6);
    items.push_back({w0, t0, o0});
    if (w1.has_value()) items.push_back({w1.value(), t1, o1.value()});
    if (w2.has_value()) items.push_back({w2.value(), t2, o2.value()});
    if (w3.has_value()) items.push_back({w3.value(), t3, o3.value()});
    if (w4.has_value()) items.push_back({w4.value(), t4, o4.value()});
    if (w5.has_value()) items.push_back({w5.value(), t5, o5.value()});

    const size_t n_ops = items.size();
    const int64_t M = x.size(0), K = x.size(1);
    const bool use_cuda = (x.device().device_type == kDLCUDA);

    std::vector<uint64_t> w_bdas(n_ops);
    size_t total = align_up(size_t(M * K) * dt_bytes(x.dtype()));
    for (size_t i = 0; i < n_ops; i++) {
        w_bdas[i] = cached_weight_bda(items[i].w);
        total += align_up(size_t(M * items[i].w.size(0)) * dt_bytes(items[i].o.dtype()));
    }

    IoArena io;
    io.begin(total, use_cuda, x.device());
    size_t x_off = io.reserve(size_t(M * K) * dt_bytes(x.dtype()));
    io.put(x_off, x);
    std::vector<size_t> y_offs(n_ops);
    for (size_t i = 0; i < n_ops; i++) {
        y_offs[i] = io.reserve(size_t(M * items[i].w.size(0)) * dt_bytes(items[i].o.dtype()));
    }

    int slot = 0;
    VkCommandBuffer cb = acquire_cmd(slot);
    const int x_dt = dl_to_dt(x.dtype());
    for (size_t i = 0; i < n_ops; i++) {
        const int64_t N = items[i].w.size(0);
        GemvPC pc{w_bdas[i], io.bda(x_off), io.bda(y_offs[i]), int32_t(K), int32_t(N), int32_t(M),
                  int32_t(items[i].ty), x_dt, dl_to_dt(items[i].o.dtype()), int32_t(N)};
        cmd_dispatch(cb, s.p_gemv, pc, uint32_t((N + 3) / 4), uint32_t(M));
    }
    submit_cmd(slot, use_cuda, x.device());
    for (size_t i = 0; i < n_ops; i++) {
        io.get_2d(y_offs[i], items[i].o, M, items[i].w.size(0));
    }
}

void op_moe(TensorView wgu, int64_t gu_ty, TensorView wdn, int64_t dn_ty,
            TensorView x, TensorView ids, TensorView weights, Optional<TensorView> scales,
            TensorView out, double gcap, double ucap) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();

    const int64_t T = ids.size(0), top_k = ids.size(1);
    const int64_t K = x.size(1), twoI = wgu.size(1), I = twoI / 2, H = wdn.size(1);
    const bool use_cuda = (x.device().device_type == kDLCUDA);

    const uint64_t wgu_bda = cached_weight_bda(wgu);
    const uint64_t wdn_bda = cached_weight_bda(wdn);
    const uint64_t sc_bda = scales.has_value() ? cached_weight_bda(scales.value()) : 0ull;

    const size_t x_bytes = size_t(T * K) * dt_bytes(x.dtype());
    const size_t ids_bytes = size_t(T * top_k) * 4;
    const size_t wt_bytes = size_t(T * top_k) * 4;
    const size_t out_bytes = size_t(T * H) * dt_bytes(out.dtype());

    IoArena io;
    io.begin(align_up(x_bytes) + align_up(ids_bytes) + align_up(wt_bytes) + align_up(out_bytes), use_cuda, x.device());
    size_t x_off = io.reserve(x_bytes);
    size_t ids_off = io.reserve(ids_bytes);
    size_t wt_off = io.reserve(wt_bytes);
    size_t out_off = io.reserve(out_bytes);
    io.put(x_off, x);
    io.put(ids_off, ids);
    io.put(wt_off, weights);

    // Device scratch for inter [T, top_k, I] fp32 and y [T, top_k, H] fp32
    const size_t inter_bytes = size_t(T * top_k * I) * 4;
    const size_t y_bytes = size_t(T * top_k * H) * 4;
    ensure_buf(s.scratch, align_up(inter_bytes) + align_up(y_bytes), false, false);
    const uint64_t inter_bda = s.scratch.bda;
    const uint64_t y_bda = s.scratch.bda + align_up(inter_bytes);

    int slot = 0;
    VkCommandBuffer cb = acquire_cmd(slot);

    // 1. Fused Gate+Up GEMV + scales + SITU
    MoeGuSituPC pc1{wgu_bda, io.bda(x_off), io.bda(ids_off), sc_bda, inter_bda,
                    int32_t(K), int32_t(I), int32_t(top_k), int32_t(T),
                    int32_t(gu_ty), dl_to_dt(x.dtype()), float(gcap), float(ucap)};
    cmd_dispatch(cb, s.p_gu_situ, pc1, uint32_t((I + 3) / 4), uint32_t(top_k), uint32_t(T));
    cmd_barrier(cb);

    // 2. Down GEMV + down scale
    MoeDnPC pc2{wdn_bda, inter_bda, io.bda(ids_off), sc_bda, y_bda,
                int32_t(I), int32_t(H), int32_t(top_k), int32_t(T),
                int32_t(dn_ty), 0, 0, 1, 4, 2};
    cmd_dispatch(cb, s.p_moe_dn, pc2, uint32_t((H + 3) / 4), uint32_t(top_k), uint32_t(T));
    cmd_barrier(cb);

    // 3. Weighted sum -> out (directly in caller's dtype!)
    WsumPC pc3{y_bda, io.bda(ids_off), io.bda(wt_off), io.bda(out_off),
               int32_t(H), int32_t(top_k), int32_t(T), 0, dl_to_dt(out.dtype())};
    cmd_dispatch(cb, s.p_wsum, pc3, uint32_t((H + 255) / 256), uint32_t(T));

    submit_cmd(slot, use_cuda, x.device());
    io.get_flat(out_off, out);
}

void op_mul_mat_id(TensorView w, int64_t ty, TensorView x, TensorView ids, TensorView out, Optional<TensorView> scale) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();

    const int64_t T = ids.size(0), top_k = ids.size(1);
    const int64_t ky = x.size(1), K = x.size(2), H = w.size(1);
    const bool use_cuda = (x.device().device_type == kDLCUDA);
    const size_t w_bytes = size_t(w.numel()) * dt_bytes(w.dtype());
    const bool stage_w = (w.device().device_type == kDLCUDA && !s.resident_banks);
    const uint64_t w_bda_cached = stage_w ? 0ull : cached_weight_bda(w);

    const size_t x_bytes = size_t(x.numel()) * dt_bytes(x.dtype());
    const size_t ids_bytes = size_t(T * top_k) * 4;
    const size_t y_bytes = size_t(T * top_k * H) * dt_bytes(out.dtype());
    const size_t sc_bytes = scale.has_value() ? size_t(scale.value().size(0)) * 4 : 0;

    IoArena io;
    io.begin((stage_w ? align_up(w_bytes) : 0) + align_up(x_bytes) + align_up(ids_bytes) + align_up(sc_bytes) + align_up(y_bytes), use_cuda, x.device());
    size_t w_off = stage_w ? io.reserve(w_bytes) : 0;
    size_t x_off = io.reserve(x_bytes);
    size_t ids_off = io.reserve(ids_bytes);
    size_t sc_off = sc_bytes ? io.reserve(sc_bytes) : 0;
    size_t y_off = io.reserve(y_bytes);
    if (stage_w) {
        auto wh = gather_to_host(w);
#ifdef FREETOKEN_VK_CUDA
        if (io.use_cuda) {
            CU_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(io.buf->cu_ptr) + w_off, wh.data(),
                                     w_bytes, cudaMemcpyHostToDevice, cu_stream(x.device())));
        } else
#endif
        {
            std::memcpy(static_cast<uint8_t *>(io.buf->mapped) + w_off, wh.data(), w_bytes);
        }
    }
    io.put(x_off, x);
    io.put(ids_off, ids);
    if (sc_bytes) {
        auto sc_host = gather_to_host(scale.value());
#ifdef FREETOKEN_VK_CUDA
        if (io.use_cuda) {
            CU_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(io.buf->cu_ptr) + sc_off, sc_host.data(),
                                     sc_bytes, cudaMemcpyHostToDevice, cu_stream(x.device())));
        } else
#endif
        {
            std::memcpy(static_cast<uint8_t *>(io.buf->mapped) + sc_off, sc_host.data(), sc_bytes);
        }
    }

    int slot = 0;
    VkCommandBuffer cb = acquire_cmd(slot);
    MoeDnPC pc{stage_w ? io.bda(w_off) : w_bda_cached, io.bda(x_off), io.bda(ids_off),
               sc_bytes ? io.bda(sc_off) : 0ull, io.bda(y_off),
               int32_t(K), int32_t(H), int32_t(top_k), int32_t(T),
               int32_t(ty), dl_to_dt(x.dtype()), dl_to_dt(out.dtype()), ky == top_k ? 1 : 0, 1, 0};
    cmd_dispatch(cb, s.p_moe_dn, pc, uint32_t((H + 3) / 4), uint32_t(top_k), uint32_t(T));
    submit_cmd(slot, use_cuda, x.device());
    io.get_flat(y_off, out);
}

void op_wsum(TensorView y, TensorView weights, TensorView out) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();
    const int64_t T = y.size(0), top_k = y.size(1), H = y.size(2);
    const bool use_cuda = (y.device().device_type == kDLCUDA);
    const size_t y_bytes = size_t(T * top_k * H) * dt_bytes(y.dtype());
    const size_t wt_bytes = size_t(T * top_k) * 4;
    const size_t out_bytes = size_t(T * H) * dt_bytes(out.dtype());
    IoArena io;
    io.begin(align_up(y_bytes) + align_up(wt_bytes) + align_up(out_bytes), use_cuda, y.device());
    size_t y_off = io.reserve(y_bytes);
    size_t wt_off = io.reserve(wt_bytes);
    size_t out_off = io.reserve(out_bytes);
    io.put(y_off, y);
    io.put(wt_off, weights);

    int slot = 0;
    VkCommandBuffer cb = acquire_cmd(slot);
    WsumPC pc{io.bda(y_off), 0ull, io.bda(wt_off), io.bda(out_off),
              int32_t(H), int32_t(top_k), int32_t(T), dl_to_dt(y.dtype()), dl_to_dt(out.dtype())};
    cmd_dispatch(cb, s.p_wsum, pc, uint32_t((H + 255) / 256), uint32_t(T));
    submit_cmd(slot, use_cuda, y.device());
    io.get_flat(out_off, out);
}

void op_get_rows(TensorView w, int64_t ty, int64_t K, TensorView ids, TensorView out) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();
    const int64_t M = ids.size(0);
    const bool use_cuda = (ids.device().device_type == kDLCUDA);
    const uint64_t w_bda = cached_weight_bda(w);
    const size_t ids_bytes = size_t(M) * 4;
    const size_t out_bytes = size_t(M * K) * dt_bytes(out.dtype());
    IoArena io;
    io.begin(align_up(ids_bytes) + align_up(out_bytes), use_cuda, ids.device());
    size_t ids_off = io.reserve(ids_bytes);
    size_t out_off = io.reserve(out_bytes);
    io.put(ids_off, ids);

    int slot = 0;
    VkCommandBuffer cb = acquire_cmd(slot);
    GetRowsPC pc{w_bda, io.bda(ids_off), io.bda(out_off), int32_t(K), int32_t(M), int32_t(ty), dl_to_dt(out.dtype())};
    cmd_dispatch(cb, s.p_get_rows, pc, uint32_t((K + 255) / 256), uint32_t(M));
    submit_cmd(slot, use_cuda, ids.device());
    io.get_flat(out_off, out);
}

void op_dequantize(TensorView w, int64_t ty, int64_t K, TensorView out) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init();
    const int64_t rows = out.size(0);
    const size_t rowb = ggml_row_bytes_for(int(ty), int(K));
    auto host = gather_to_host(w);
    const bool use_cuda = (out.device().device_type == kDLCUDA);
    const int64_t chunk = 8192;
    const size_t out_esz = dt_bytes(out.dtype());
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, rows - r0);
        const size_t wb = size_t(nr) * rowb;
        const size_t ob = size_t(nr * K) * out_esz;
        IoArena io;
        io.begin(align_up(wb) + align_up(ob), use_cuda, out.device());
        size_t w_off = io.reserve(wb);
        size_t o_off = io.reserve(ob);
#ifdef FREETOKEN_VK_CUDA
        if (io.use_cuda) {
            CU_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(io.buf->cu_ptr) + w_off, host.data() + r0 * rowb,
                                     wb, cudaMemcpyHostToDevice, cu_stream(out.device())));
        } else
#endif
        {
            std::memcpy(static_cast<uint8_t *>(io.buf->mapped) + w_off, host.data() + r0 * rowb, wb);
        }
        int slot = 0;
        VkCommandBuffer cb = acquire_cmd(slot);
        GetRowsPC pc{io.bda(w_off), 0ull, io.bda(o_off), int32_t(K), int32_t(nr), int32_t(ty), dl_to_dt(out.dtype())};
        cmd_dispatch(cb, s.p_get_rows, pc, uint32_t((K + 255) / 256), uint32_t(nr));
        submit_cmd(slot, use_cuda, out.device());
#ifdef FREETOKEN_VK_CUDA
        if (io.use_cuda) {
            CU_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(out.data_ptr()) + size_t(r0 * K) * out_esz,
                                     static_cast<const uint8_t *>(io.buf->cu_ptr) + o_off,
                                     ob, cudaMemcpyDeviceToDevice, cu_stream(out.device())));
        } else if (out.device().device_type == kDLCUDA) {
            CU_CHECK(cudaSetDevice(out.device().device_id));
            CU_CHECK(cudaMemcpy(static_cast<uint8_t *>(out.data_ptr()) + size_t(r0 * K) * out_esz,
                                static_cast<const uint8_t *>(io.buf->mapped) + o_off,
                                ob, cudaMemcpyHostToDevice));
        } else
#endif
        {
            std::memcpy(static_cast<uint8_t *>(out.data_ptr()) + size_t(r0 * K) * out_esz,
                        static_cast<const uint8_t *>(io.buf->mapped) + o_off, ob);
        }
    }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_init, op_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_set_bank_mode, op_set_bank_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_mul_mat, op_mul_mat);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_mul_mat_batch, op_mul_mat_batch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_moe, op_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_mul_mat_id, op_mul_mat_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_wsum, op_wsum);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_get_rows, op_get_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_dequantize, op_dequantize);
