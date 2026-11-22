// ik2-vk: Inkling-2's routed-expert FFN as dedicated Vulkan compute shaders, in the
// style of maya's ik2 (arithmetic of its ik2_exact.h / transformers' modeling_inkling2):
// one command buffer per call -- gate_up GEMV -> SITU -> down GEMV per routed expert,
// then the weighted sum -- one submit and one fence, intermediates stay on the device.
// The packed expert banks upload once into resident device buffers (fused serving keeps
// them in host RAM); only the activations cross the host per call.

#include <vulkan/vulkan.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "ik2_shaders.h"

#ifdef FREETOKEN_VK_CUDA
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#endif

using tvm::ffi::TensorView;

namespace {

#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) throw std::runtime_error("ik2_vk: " #x " failed"); } while (0)

struct PC {
    int32_t K, N, ty, row0, x0, y0, te, e, half_n, dim2;
    float gcap, ucap;
};

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    void * mapped = nullptr;
    size_t size = 0;
};

struct State {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet ds[6] = {};  // deq_gu, deq_dn, gemv_gu, gemv_dn, situ, wsum
    VkPipeline deq = VK_NULL_HANDLE, gemv = VK_NULL_HANDLE, situ = VK_NULL_HANDLE, wsum = VK_NULL_HANDLE;
    Buf xg, pos, wt, scg, eidx, eids, out;   // host-visible
    Buf gu, inter, y, wf_gu, wf_dn;    // device scratch
    Buf stage;                  // device-bank upload staging
    std::unordered_map<uint64_t, Buf> banks;
    std::mutex mu;
    bool ready = false;
};

State & st() {
    static State s;
    return s;
}

uint32_t mem_type(uint32_t bits, VkFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(st().phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("ik2_vk: no suitable memory type");
}

Buf make_buf(size_t size, bool host) {
    auto & s = st();
    Buf bf;
    bf.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(s.dev, &bi, nullptr, &bf.b));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(s.dev, bf.b, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mem_type(mr.memoryTypeBits,
        host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(s.dev, &ai, nullptr, &bf.m));
    VK_CHECK(vkBindBufferMemory(s.dev, bf.b, bf.m, 0));
    if (host) VK_CHECK(vkMapMemory(s.dev, bf.m, 0, size, 0, &bf.mapped));
    return bf;
}

void free_buf(Buf & bf) {
    if (bf.b) vkDestroyBuffer(st().dev, bf.b, nullptr);
    if (bf.m) vkFreeMemory(st().dev, bf.m, nullptr);
    bf = Buf{};
}

void grow(Buf & bf, size_t size, bool host) {
    if (bf.size >= size) return;
    free_buf(bf);
    bf = make_buf(std::max(size, bf.size * 2), host);
}

VkShaderModule module(const uint32_t * code, size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(st().dev, &ci, nullptr, &m));
    return m;
}

VkPipeline pipeline(VkShaderModule m) {
    VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    si.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    si.module = m;
    si.pName = "main";
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = si;
    ci.layout = st().pl;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(st().dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
    vkDestroyShaderModule(st().dev, m, nullptr);
    return p;
}

void init(int device) {
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
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(s.inst, &n, devs.data());
    if (device < 0 || device >= int(n)) throw std::runtime_error("ik2_vk: no such Vulkan device");
    s.phys = devs[device];
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &qn, qp.data());
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    bool found = false;
    for (uint32_t i = 0; i < qn; i++) {
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            s.qfam = i;
            qci.queueFamilyIndex = i;
            qci.queueCount = 1;
            qci.pQueuePriorities = &prio;
            found = true;
            break;
        }
    }
    if (!found) throw std::runtime_error("ik2_vk: no compute queue");
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(s.phys, &dci, nullptr, &s.dev));
    vkGetDeviceQueue(s.dev, s.qfam, 0, &s.queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = s.qfam;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(s.dev, &pci, nullptr, &s.pool));
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = s.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(s.dev, &cbi, &s.cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(s.dev, &fci, nullptr, &s.fence));

    VkDescriptorSetLayoutBinding binds[10];
    for (int i = 0; i < 10; i++) {
        binds[i] = {};
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 10;
    dlci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(s.dev, &dlci, nullptr, &s.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &s.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(s.dev, &plci, nullptr, &s.pl));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 60};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 6;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(s.dev, &dpci, nullptr, &s.dpool));
    VkDescriptorSetLayout layouts[6] = {s.dsl, s.dsl, s.dsl, s.dsl, s.dsl, s.dsl};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = s.dpool;
    dsai.descriptorSetCount = 6;
    dsai.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(s.dev, &dsai, s.ds));

    s.deq = pipeline(module(ik2_dequant_spv, ik2_dequant_spv_len));
    s.gemv = pipeline(module(ik2_gemv_spv, ik2_gemv_spv_len));
    s.situ = pipeline(module(ik2_situ_spv, ik2_situ_spv_len));
    s.wsum = pipeline(module(ik2_wsum_spv, ik2_wsum_spv_len));
    s.ready = true;
}

// host-visible input upload: a plain memcpy, no submit
void put(Buf & bf, const void * data, size_t bytes, size_t offset = 0) {
    grow(bf, offset + bytes, true);
    std::memcpy(static_cast<uint8_t *>(bf.mapped) + offset, data, bytes);
}

void write_set(int idx, std::initializer_list<std::pair<int, Buf *>> entries) {
    auto & s = st();
    VkDescriptorBufferInfo infos[10];
    VkWriteDescriptorSet writes[10];
    int n = 0;
    bool used[10] = {};
    for (auto & [binding, bf] : entries) {
        infos[n] = {bf->b, 0, bf->size};
        writes[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[n].dstSet = s.ds[idx];
        writes[n].dstBinding = binding;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[n].pBufferInfo = &infos[n];
        used[binding] = true;
        n++;
    }
    // unused bindings still need a valid descriptor
    static Buf dummy;
    if (!dummy.b) dummy = make_buf(16, true);
    for (int b = 0; b < 10; b++) {
        if (used[b]) continue;
        infos[n] = {dummy.b, 0, dummy.size};
        writes[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[n].dstSet = s.ds[idx];
        writes[n].dstBinding = b;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[n].pBufferInfo = &infos[n];
        n++;
    }
    vkUpdateDescriptorSets(s.dev, n, writes, 0, nullptr);
}

void barrier() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(st().cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void dispatch(VkPipeline p, int set, const PC & pc, uint32_t groups) {
    auto & s = st();
    vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &s.ds[set], 0, nullptr);
    vkCmdPushConstants(s.cmd, s.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
    vkCmdDispatch(s.cmd, groups, 1, 1);
}

// content samples fingerprint a bank: a freed tensor's pointer can be reused by a
// different allocation, so the pointer alone is not a safe cache key
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

// resident packed bank: upload once, keyed by pointer + content samples
Buf & bank(TensorView w) {
    auto & s = st();
    if (w.device().device_type != kDLCPU) {
        throw std::runtime_error("ik2_vk: expert banks must be host tensors (serve GGUF experts with --moe-strategy fused)");
    }
    size_t bytes = w.numel();
    const uint8_t * base = static_cast<const uint8_t *>(w.data_ptr());
    uint64_t key = reinterpret_cast<uint64_t>(w.data_ptr());
    key ^= sample_hash(base, bytes);
    key ^= sample_hash(base + bytes / 2, bytes - bytes / 2) * 31;
    key ^= sample_hash(base + (bytes > 64 ? bytes - 64 : 0), bytes > 64 ? 64 : bytes) * 131;
    auto it = s.banks.find(key);
    if (it != s.banks.end()) return it->second;
    Buf bf = make_buf(bytes, false);
    grow(s.stage, bytes, true);
    std::memcpy(s.stage.mapped, w.data_ptr(), bytes);
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = s.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(s.dev, &cbi, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, s.stage.b, bf.b, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(s.queue, 1, &si, s.fence));
    VK_CHECK(vkWaitForFences(s.dev, 1, &s.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(s.dev, 1, &s.fence));
    vkFreeCommandBuffers(s.dev, s.pool, 1, &cb);
    return s.banks.emplace(key, bf).first->second;
}

std::vector<uint8_t> to_host(TensorView t) {
    size_t n = t.numel() * (t.dtype().bits / 8);
    std::vector<uint8_t> h(n);
    if (n == 0) return h;
#ifdef FREETOKEN_VK_CUDA
    if (t.device().device_type == kDLCUDA) {
        cudaSetDevice(t.device().device_id);
        auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(t.device().device_type, t.device().device_id));
        cudaStreamSynchronize(stream == nullptr ? cudaStreamLegacy : stream);
        cudaMemcpy(h.data(), t.data_ptr(), n, cudaMemcpyDeviceToHost);
        return h;
    }
#endif
    std::memcpy(h.data(), t.data_ptr(), n);
    return h;
}

void op_init(int64_t device) {
    std::lock_guard<std::mutex> lock(st().mu);
    init(int(device));
}

void op_moe(TensorView wgu, int64_t gu_ty, TensorView wdn, int64_t dn_ty,
            TensorView x, TensorView ids, TensorView weights, TensorView scales,
            TensorView out, double gcap, double ucap) {
    auto & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    init(0);
    const int T = int(ids.size(0)), k = int(ids.size(1));
    const int K = int(x.size(1)), twoI = int(wgu.size(1)), I = twoI / 2, H = int(wdn.size(1)), E = int(wgu.size(0));

    auto xh = to_host(x);
    auto idh = to_host(ids);
    auto wh = to_host(weights);
    auto sch = to_host(scales);
    // activations arrive in the model dtype; the shaders run F32
    std::vector<float> xf32(size_t(T) * K);
    if (x.dtype().code == kDLFloat && x.dtype().bits == 32) {
        std::memcpy(xf32.data(), xh.data(), xf32.size() * 4);
    } else if (x.dtype().code == kDLBfloat) {
        const uint16_t * p = reinterpret_cast<const uint16_t *>(xh.data());
        for (size_t i = 0; i < xf32.size(); i++) {
            uint32_t u = uint32_t(p[i]) << 16;
            std::memcpy(&xf32[i], &u, 4);
        }
    } else {
        const uint16_t * p = reinterpret_cast<const uint16_t *>(xh.data());
        for (size_t i = 0; i < xf32.size(); i++) {
            uint32_t h = p[i], sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
            uint32_t u = e == 0 ? (sign | (m ? 0x38000000u : 0u)) : sign | ((e + 112) << 23) | (m << 13);
            if (e == 0 && m) { float f; uint32_t uu = sign; std::memcpy(&f, &uu, 4); xf32[i] = f + float(m) * 5.9604644775390625e-8f; continue; }
            if (e == 31) u = sign | 0x7f800000u | (m << 13);
            std::memcpy(&xf32[i], &u, 4);
        }
    }
    const float * xf = xf32.data();
    const int32_t * idf = reinterpret_cast<const int32_t *>(idh.data());
    const float * wf = reinterpret_cast<const float *>(wh.data());

    // group the valid routes by expert
    std::vector<int> uniq;
    for (int i = 0; i < T * k; i++) if (idf[i] >= 0) uniq.push_back(idf[i]);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::vector<std::vector<int>> pos(uniq.size());
    for (int i = 0; i < T * k; i++) {
        if (idf[i] < 0) continue;
        int e = int(std::lower_bound(uniq.begin(), uniq.end(), idf[i]) - uniq.begin());
        pos[e].push_back(i);
    }
    // route positions in expert-group order: the shaders read each expert's rows contiguous
    std::vector<int> route_pos(T * k, -1);
    int R = 0;
    for (size_t e = 0; e < uniq.size(); e++) {
        for (int i : pos[e]) route_pos[i] = R++;
    }

    // host-visible inputs: gathered activations in route order, positions, weights, scales
    std::vector<float> xg(size_t(R) * K);
    for (size_t e = 0; e < uniq.size(); e++) {
        for (size_t j = 0; j < pos[e].size(); j++) {
            int i = pos[e][j];
            std::memcpy(xg.data() + size_t(route_pos[i]) * K, xf + size_t(i / k) * K, K * 4);
        }
    }
    put(s.xg, xg.data(), xg.size() * 4);
    put(s.pos, route_pos.data(), route_pos.size() * 4);
    put(s.wt, wf, size_t(T) * k * 4);
    grow(s.out, size_t(T) * H * 4, true);

    // device scratch; the wf buffers hold a chunk of experts' decoded rows
    const int CHUNK = 8;
    grow(s.gu, size_t(R) * twoI * 4, false);
    grow(s.inter, size_t(R) * I * 4, false);
    grow(s.y, size_t(R) * H * 4, false);
    grow(s.wf_gu, size_t(CHUNK) * twoI * K * 4, false);
    grow(s.wf_dn, size_t(CHUNK) * H * I * 4, false);

    // per-route expert index (within the whole route list) and gathered scales
    std::vector<int32_t> eidx(R);
    std::vector<float> scg(size_t(R) * 4, 1.0f);
    const float * scf = reinterpret_cast<const float *>(sch.data());
    {
        int r = 0;
        for (size_t e = 0; e < uniq.size(); e++) {
            for (size_t j = 0; j < pos[e].size(); j++) {
                eidx[r] = int(e);
                if (!sch.empty()) std::memcpy(scg.data() + size_t(r) * 4, scf + size_t(uniq[e]) * 4, 16);
                r++;
            }
        }
    }
    put(s.eidx, eidx.data(), eidx.size() * 4);
    put(s.scg, scg.data(), scg.size() * 4);
    // each chunk's actual expert ids, concatenated (the dequant reads them by offset)
    std::vector<int32_t> eids_all;
    std::vector<int> eids_off;
    for (size_t e0 = 0; e0 < uniq.size(); e0 += CHUNK) {
        eids_off.push_back(int(eids_all.size()));
        const size_t e1 = std::min(uniq.size(), e0 + CHUNK);
        for (size_t e = e0; e < e1; e++) eids_all.push_back(uniq[e]);
    }
    put(s.eids, eids_all.data(), eids_all.size() * 4);

    Buf & bgu = bank(wgu);
    Buf & bdn = bank(wdn);
    write_set(0, {{0, &bgu}, {1, &s.eids}, {3, &s.wf_gu}});
    write_set(1, {{0, &bdn}, {1, &s.eids}, {3, &s.wf_dn}});
    write_set(2, {{0, &s.wf_gu}, {1, &s.eidx}, {2, &s.xg}, {3, &s.gu}, {8, &s.scg}});
    write_set(3, {{0, &s.wf_dn}, {1, &s.eidx}, {2, &s.inter}, {3, &s.y}, {8, &s.scg}});
    write_set(4, {{3, &s.gu}, {4, &s.inter}, {8, &s.scg}});
    write_set(5, {{5, &s.y}, {6, &s.pos}, {7, &s.wt}, {9, &s.out}});

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(s.cmd, &bi));
    int off = 0, ci = 0;
    for (size_t e0 = 0; e0 < uniq.size(); e0 += CHUNK, ci++) {
        const size_t e1 = std::min(uniq.size(), e0 + CHUNK);
        const int ce = int(e1 - e0);
        int routes = 0;
        for (size_t e = e0; e < e1; e++) routes += int(pos[e].size());
        PC pc{};
        pc.te = routes; pc.gcap = float(gcap); pc.ucap = float(ucap); pc.y0 = off;
        // decode the chunk's gate_up rows, then one GEMV over all its routes
        pc.K = K; pc.N = twoI; pc.ty = int(gu_ty); pc.e = ce; pc.row0 = eids_off[ci];
        dispatch(s.deq, 0, pc, (uint32_t(ce) * twoI * K + 255) / 256);
        barrier();
        pc.row0 = int(e0); pc.x0 = 0; pc.half_n = -1;
        dispatch(s.gemv, 2, pc, (uint32_t(routes) * twoI + 255) / 256);
        barrier();
        pc.dim2 = I;
        dispatch(s.situ, 4, pc, (uint32_t(routes) * I + 255) / 256);
        barrier();
        // decode the chunk's down rows, then the GEMV
        pc.K = I; pc.N = H; pc.ty = int(dn_ty); pc.e = ce; pc.row0 = eids_off[ci]; pc.dim2 = 0;
        dispatch(s.deq, 1, pc, (uint32_t(ce) * H * I + 255) / 256);
        barrier();
        pc.N = H; pc.row0 = int(e0); pc.half_n = 0;
        dispatch(s.gemv, 3, pc, (uint32_t(routes) * H + 255) / 256);
        barrier();
        off += routes;
    }
    PC pc{};
    pc.te = T; pc.e = k; pc.dim2 = H;
    dispatch(s.wsum, 5, pc, (uint32_t(T) * H + 255) / 256);
    VK_CHECK(vkEndCommandBuffer(s.cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd;
    VK_CHECK(vkQueueSubmit(s.queue, 1, &si, s.fence));
    VK_CHECK(vkWaitForFences(s.dev, 1, &s.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(s.dev, 1, &s.fence));

    // out: fp32 -> the caller's dtype/device
    const float * res = static_cast<const float *>(s.out.mapped);
    size_t n = size_t(T) * H;
    if (out.dtype().code == kDLFloat && out.dtype().bits == 32) {
#ifdef FREETOKEN_VK_CUDA
        if (out.device().device_type == kDLCUDA) {
            cudaSetDevice(out.device().device_id);
            cudaMemcpy(out.data_ptr(), res, n * 4, cudaMemcpyHostToDevice);
            return;
        }
#endif
        std::memcpy(out.data_ptr(), res, n * 4);
        return;
    }
    // bf16 / f16 out
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        std::memcpy(&u, &res[i], 4);
        if (out.dtype().code == kDLBfloat) {
            u += 0x7fff + ((u >> 16) & 1);
            h[i] = uint16_t(u >> 16);
        } else {
            uint32_t sign = (u >> 16) & 0x8000;
            int32_t e = int32_t((u >> 23) & 0xff) - 127 + 15;
            uint32_t m = u & 0x7fffff;
            uint16_t v;
            if (e >= 31) v = uint16_t(sign | 0x7c00);
            else if (e <= 0) {
                if (e < -10) v = uint16_t(sign);
                else {
                    m |= 0x800000;
                    uint32_t sh = uint32_t(14 - e), q = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
                    if (rem > half || (rem == half && (q & 1))) q++;
                    v = uint16_t(sign | q);
                }
            } else {
                uint32_t q = m >> 13, rem = m & 0x1fff;
                if (rem > 0x1000 || (rem == 0x1000 && (q & 1))) { q++; if (q == 0x400) { q = 0; e++; } }
                v = e >= 31 ? uint16_t(sign | 0x7c00) : uint16_t(sign | (uint32_t(e) << 10) | q);
            }
            h[i] = v;
        }
    }
#ifdef FREETOKEN_VK_CUDA
    if (out.device().device_type == kDLCUDA) {
        cudaSetDevice(out.device().device_id);
        cudaMemcpy(out.data_ptr(), h.data(), n * 2, cudaMemcpyHostToDevice);
        return;
    }
#endif
    std::memcpy(out.data_ptr(), h.data(), n * 2);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_init, op_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ik2_vk_moe, op_moe);
