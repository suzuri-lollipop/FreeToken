// C++ radix prefix-tree core for the three sibling caches (RadixPrefixCache,
// SWARadixCache, HybridRadixCache). A statement-level port of the Python trees in
// python/freetoken/kvcache/{radix_cache,swa_radix_cache,hybrid_radix_cache}.py --
// the Python classes remain the reference and the fallback (FREETOKEN_RADIX_BACKEND).
//
// What moves to C++: the tree walk (bucket lookup + key mismatch + page-aligned
// split), lock/unlock chains, both eviction passes, counters and the LRU clock.
// What stays in torch: node values are the caller's own int32 slot-index tensors
// (a GPU page-table row slice in production, cloned at insert), so path
// concatenation is at::cat on the original device with zero host copies.
//
// Identity contract (tests/kvcache/radix): split_at keeps the ORIGINAL node as the
// suffix and mints a new prefix node, so handles issued before a split still name
// the same slots. Node identity for Python is a stable per-id NodeRef object from
// a strong registry pruned on unlink; ids carry a generation tag so a stale id is
// a loud error, never a silent alias of a recycled slot.
//
// LRU clock: a per-tree monotonic counter replaces time.monotonic_ns(). One tick
// per stamping event (walk / new node), the same order semantics, no syscall, and
// no cross-call ties. SWA keeps the Python _EVENT_STRIDE scheme (path stamps
// strictly decreasing toward the root) on the same counter.
//
// Scheduler-thread only (like the Python trees): no internal locking.

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

enum class Mode : int { Plain = 0, SWA = 1, Hybrid = 2 };

// SWARadixCache._EVENT_STRIDE: one logical event leaves room for per-node depth
// offsets below it (strictly decreasing toward root) without colliding.
constexpr int64_t kEventStride = 1 << 24;

inline int64_t align_down_i(int64_t x, int64_t p) { return (x / p) * p; }

// splitmix64-style mix over the first-page tokens: the children bucket hash.
inline uint64_t hash_page(const int64_t *toks, int64_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
  for (int64_t i = 0; i < n; ++i) {
    h ^= (uint64_t)toks[i] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  }
  return h;
}

struct Node {
  std::vector<int64_t> key;   // token ids of this node's span (host copy)
  torch::Tensor value;        // per-token slot indices; any device; opaque to the walk
  int64_t parent = -1;        // arena slot of the parent; the root's parent is -1
  // first-page hash -> candidate child slots. Bucket keys are unique per parent
  // (the walk would have matched instead of inserting a second), so the vector is
  // virtually always size 1; hash collisions resolve by a full key compare.
  std::unordered_map<uint64_t, std::vector<int32_t>> children;
  int32_t ref_count = 0;
  int64_t stamp = 0;          // LRU timestamp (logical clock)
  int32_t mamba_value = -1;   // hybrid: GDN snapshot slot; -1 is Python's None
  int32_t mamba_ref = 0;
  bool swa_tomb = false;      // swa: window KV freed, full KV survives
  int32_t swa_ref = 0;
  int64_t swa_uuid = -1;      // swa: window-boundary lock handle; -1 is None
  int64_t uuid = 0;           // debug: mirrors RadixTreeNode.counter
  uint32_t gen = 0;           // bumped on slot reuse; part of the external id
  bool dead = false;          // unlinked from the tree (evicted)
};

class TreeCore;

// Python-visible node identity. Holds a WEAK tree pointer: the facade owns the
// tree and the tree owns the (strong) NodeRef registry, so a strong back-pointer
// here would form an unbreakable shared_ptr cycle. Accessors fail loudly on a
// destroyed tree or a stale (recycled-slot) id.
class NodeRef {
 public:
  NodeRef(std::weak_ptr<TreeCore> tree, int64_t id) : tree_(std::move(tree)), id_(id) {}
  int64_t id() const { return id_; }
  std::shared_ptr<TreeCore> tree() const;

 private:
  std::weak_ptr<TreeCore> tree_;
  int64_t id_;
};

// Eviction heap entry: (stamp, slot) min-ordered. Ties break on slot -- any
// min-stamp victim is a legal LRU pick (the reference model demands exactly that).
using HeapEntry = std::pair<int64_t, int32_t>;
using EvictHeap = std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>>;

class TreeCore : public std::enable_shared_from_this<TreeCore> {
 public:
  TreeCore(int mode, int64_t page_size, int64_t window, std::string device)
      : mode_((Mode)mode), page_(page_size), window_(window) {
    TORCH_CHECK(page_size >= 1, "page_size must be >= 1");
    TORCH_CHECK(mode_ == Mode::Plain || mode_ == Mode::SWA || mode_ == Mode::Hybrid,
                "unknown radix tree mode");
    if (mode_ == Mode::SWA) TORCH_CHECK(window > 0, "SWA tree requires a positive window");
    opts_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(device));
    empty_ = torch::empty({0}, opts_);
    // The root: slot 0, permanent, always protected (ref_count 1) like all three
    // Python classes. Value/key stay empty (nothing reads the root's fields).
    nodes_.emplace_back();
    gen_ctr_.push_back(0);
    Node &r = nodes_[0];
    r.uuid = uuid_ctr_++;
    r.ref_count = 1;
    r.value = empty_;
    root_ = 0;
  }

  // ------------------------------------------------------------------ ids
  static int64_t make_id(int32_t slot, uint32_t gen) {
    return (int64_t)(uint32_t)slot | ((int64_t)gen << 32);
  }
  static int32_t slot_of(int64_t id) { return (int32_t)(uint32_t)(id & 0xFFFFFFFFll); }
  Node &node(int64_t id) {
    int32_t s = slot_of(id);
    TORCH_CHECK(s >= 0 && (size_t)s < nodes_.size(), "radix: node id out of range");
    Node &n = nodes_[s];
    TORCH_CHECK(!n.dead && n.gen == (uint32_t)((uint64_t)id >> 32),
                "radix: stale node id (the node was evicted)");
    return n;
  }
  int64_t root_id() const { return make_id(root_, nodes_[root_].gen); }

  // Stable Python identity: one NodeRef object per live id, strongly cached so
  // `a is b` / `id(a)` semantics match the Python tree objects; pruned on unlink.
  py::object ref(int64_t id) {
    auto it = refs_.find(id);
    if (it != refs_.end()) return it->second;
    py::object obj = py::cast(NodeRef(weak_from_this(), id));
    refs_.emplace(id, obj);
    return obj;
  }
  void prune_ref(int64_t id) { refs_.erase(id); }

  // ------------------------------------------------------------------ clock
  int64_t tic() { return ++clk_; }                        // plain/hybrid event tick
  int64_t stride_tic() { return (++clk_) * kEventStride; }  // swa event tick

  // ------------------------------------------------------------------ key helpers
  uint64_t node_hash(const Node &n) const {
    int64_t k = std::min(page_, (int64_t)n.key.size());
    return hash_page(n.key.data(), k);
  }

  int32_t lookup_child(int32_t pslot, const int64_t *q, int64_t qlen) const {
    // A short trailing query (qlen < page) can never match a full-page bucket --
    // the Python key_fn produces a short tuple that equals no child key.
    if (qlen < page_) return -1;
    const Node &p = nodes_[pslot];
    auto it = p.children.find(hash_page(q, page_));
    if (it == p.children.end()) return -1;
    for (int32_t cid : it->second) {
      const Node &c = nodes_[cid];
      if ((int64_t)c.key.size() >= page_ &&
          std::equal(c.key.begin(), c.key.begin() + page_, q)) {
        return cid;
      }
    }
    return -1;
  }

  void register_child(int32_t pslot, int32_t cid) {
    nodes_[pslot].children[node_hash(nodes_[cid])].push_back(cid);
  }

  // fast_compare_key parity: first differing position, capped at the shorter len.
  static int64_t mismatch_len(const std::vector<int64_t> &key, const int64_t *q, int64_t qlen) {
    int64_t n = std::min((int64_t)key.size(), qlen);
    int64_t i = 0;
    while (i < n && key[i] == q[i]) ++i;
    return i;
  }

  // ------------------------------------------------------------------ node ops
  int32_t new_node(int64_t stamp) {
    int32_t slot;
    if (!free_slots_.empty()) {
      slot = free_slots_.back();
      free_slots_.pop_back();
      nodes_[slot] = Node{};  // reset every field, then a fresh generation
      nodes_[slot].gen = ++gen_ctr_[slot];
    } else {
      slot = (int32_t)nodes_.size();
      nodes_.emplace_back();
      gen_ctr_.push_back(0);
    }
    Node &n = nodes_[slot];
    n.stamp = stamp;
    n.uuid = uuid_ctr_++;
    return slot;
  }

  void set_key_value(Node &n, const int64_t *ids, int64_t idn, torch::Tensor value) {
    TORCH_CHECK(idn == value.numel(), "radix: key/value length mismatch");
    n.key.assign(ids, ids + idn);
    n.value = std::move(value);
  }

  // RadixTreeNode.split_at parity: the ORIGINAL node mutates into the suffix (its
  // id survives for already-issued handles); a NEW node becomes the root-side
  // prefix. Field migration is the union all three classes rely on: ref/swa_ref/
  // tombstone copy to both halves, swa_uuid migrates root-side, the GDN snapshot
  // stays on the suffix, the prefix inherits the LRU stamp. The parent's bucket
  // key (first page) is invariant across the split, so the bucket entry is
  // rewritten in place: slot -> prefix slot.
  int32_t split_at(int32_t slot, int64_t pos) {
    int64_t len = (int64_t)nodes_[slot].key.size();
    TORCH_CHECK(pos > 0 && pos < len,
                "radix: split point must land strictly inside the node");
    int64_t pid = nodes_[slot].parent;
    uint64_t bucket = node_hash(nodes_[slot]);  // first-page hash (split-invariant)
    int64_t orig_stamp = nodes_[slot].stamp;
    int32_t pslot = new_node(orig_stamp);
    Node &pre = nodes_[pslot];
    Node &suf = nodes_[slot];  // re-bind (new_node may have grown the deque)
    pre.key.assign(suf.key.begin(), suf.key.begin() + pos);
    pre.value = suf.value.narrow(0, 0, pos);
    pre.ref_count = suf.ref_count;
    pre.swa_ref = suf.swa_ref;
    pre.swa_tomb = suf.swa_tomb;
    pre.swa_uuid = suf.swa_uuid;
    suf.swa_uuid = -1;
    suf.key.erase(suf.key.begin(), suf.key.begin() + pos);
    suf.value = suf.value.narrow(0, pos, len - pos);
    // relink: parent's bucket now names the prefix; the suffix moves under it.
    auto it = nodes_[pid].children.find(bucket);
    TORCH_CHECK(it != nodes_[pid].children.end(),
                "radix: split target not registered under its page key");
    auto &v = it->second;
    auto at = std::find(v.begin(), v.end(), slot);
    TORCH_CHECK(at != v.end(), "radix: split target missing from its bucket");
    *at = pslot;
    pre.parent = pid;
    suf.parent = pslot;
    register_child(pslot, slot);
    return pslot;
  }

  bool is_leaf(int32_t slot) const { return nodes_[slot].children.empty(); }

  int64_t path_len(int32_t slot) const {
    int64_t total = 0;
    int32_t s = slot;
    while (s != root_) {
      total += (int64_t)nodes_[s].key.size();
      s = nodes_[s].parent;
    }
    return total;
  }

  // torch.cat of the root->node value chain (RadixCacheHandle.get_matched_indices
  // / _collect_kv). Values keep their device; an empty path yields the empty
  // tensor instead of the Python torch.cat([]) raise (callers guard on len 0).
  // A single-node path clones instead of aliasing the live node's tensor: the
  // Python torch.cat always copies, and a caller mutating the result must not be
  // able to corrupt the tree. (Eviction returns skip the clone -- those nodes are
  // dead and the tensor is a ownership hand-off.)
  torch::Tensor path_kv(int32_t slot) {
    std::vector<torch::Tensor> chain;
    int32_t s = slot;
    while (s != root_) {
      chain.push_back(nodes_[s].value);
      s = nodes_[s].parent;
    }
    if (chain.empty()) return empty_;
    if (chain.size() == 1) return chain[0].clone();
    std::reverse(chain.begin(), chain.end());
    return torch::cat(chain);
  }
  torch::Tensor path_kv_id(int64_t id) {
    node(id);
    return path_kv(slot_of(id));
  }

  // ------------------------------------------------------------------ counters
  // plain: evictable_/protected_ are THE size_info pair; swa/hybrid: the full_*
  // pair, with the second currency alongside. Same increments as the Python.
  int64_t evictable_ = 0, protected_ = 0;
  int64_t swa_evictable_ = 0, swa_protected_ = 0;
  int64_t mamba_evictable_ = 0, mamba_protected_ = 0;
  int64_t revives_ = 0;  // swa observability (SWARadixCache._revives)

  py::tuple counters() const {
    return py::make_tuple(evictable_, protected_, swa_evictable_, swa_protected_,
                          mamba_evictable_, mamba_protected_, revives_);
  }

  // ==================================================================
  // plain + hybrid walk (RadixPrefixCache._tree_walk / HybridRadixCache._walk)
  // ==================================================================
  // Returns (slot, prefix_len). Stamps every matched node (and a split's prefix
  // half) with ONE tic. `ids` borrows the caller's widened host buffer.
  std::pair<int32_t, int64_t> walk(const int64_t *ids, int64_t n) {
    int64_t prefix_len = 0;
    int32_t slot = root_;
    int64_t stamp = tic();
    while (prefix_len < n) {
      const int64_t *q = ids + prefix_len;
      int64_t qlen = n - prefix_len;
      int32_t cid = lookup_child(slot, q, qlen);
      if (cid < 0) return {slot, prefix_len};
      int64_t ml = align_down_i(mismatch_len(nodes_[cid].key, q, qlen), page_);
      // bucket hit => the first page matched => ml >= page (the Python NOTE).
      prefix_len += ml;
      if (ml != (int64_t)nodes_[cid].key.size()) {
        int32_t pslot = split_at(cid, ml);
        nodes_[pslot].stamp = stamp;
        return {pslot, prefix_len};
      }
      nodes_[cid].stamp = stamp;
      slot = cid;
    }
    return {slot, prefix_len};
  }

  // ==================================================================
  // PLAIN (RadixPrefixCache)
  // ==================================================================
  py::tuple match_plain(const torch::Tensor &ids) {
    IdsBuf b = ids_buf(ids);
    auto [slot, plen] = walk(b.ptr, b.n);
    return py::make_tuple(ref(make_id(slot, nodes_[slot].gen)), plen);
  }

  py::tuple insert_plain(const torch::Tensor &ids, const torch::Tensor &indices) {
    IdsBuf b = ids_buf(ids);
    int64_t insert_len = align_down_i(b.n, page_);
    auto [slot, plen] = walk(b.ptr, insert_len);
    if (plen != insert_len) {
      int32_t ns = new_node(tic());
      Node &nn = nodes_[ns];
      set_key_value(nn, b.ptr + plen, insert_len - plen,
                    indices.narrow(0, plen, insert_len - plen).clone());
      nn.parent = slot;
      register_child(slot, ns);
      evictable_ += (int64_t)nn.key.size();
      slot = ns;
    }
    return py::make_tuple(plen, insert_len, ref(make_id(slot, nodes_[slot].gen)));
  }

  void lock_plain(int64_t id, bool unlock) {
    node(id);
    full_chain_lock(slot_of(id), unlock);
  }

  torch::Tensor evict_plain(int64_t size) {
    if (size == 0) return empty_;
    TORCH_CHECK(size <= evictable_, "Cannot evict ", size, ", only ", evictable_,
                " is evictable");
    EvictHeap heap;
    collect_unlocked_leaves(heap);
    std::vector<torch::Tensor> out;
    std::vector<int32_t> unlinked;
    int64_t evicted = 0;
    while (evicted < size) {
      TORCH_CHECK(!heap.empty(), "Cannot evict enough cache, need ", size, ", only ",
                  evicted, " evicted");
      auto [stamp, slot] = heap.top();
      heap.pop();
      (void)stamp;
      Node &n = nodes_[slot];
      if (n.dead || n.ref_count != 0 || !is_leaf(slot) || slot == root_) continue;
      evicted += (int64_t)n.key.size();
      out.push_back(n.value);
      evictable_ -= (int64_t)n.key.size();
      int32_t parent = n.parent;
      unlink(slot, unlinked);
      // NOTE: root is always protected (ref_count 1), so it never qualifies.
      if (is_leaf(parent) && nodes_[parent].ref_count == 0 && parent != root_) {
        heap.emplace(nodes_[parent].stamp, parent);
      }
    }
    finish_unlinked(unlinked);
    return cat_or_empty(out);
  }

  // ==================================================================
  // HYBRID (HybridRadixCache)
  // ==================================================================
  py::tuple match_hybrid(const torch::Tensor &ids) {
    IdsBuf b = ids_buf(ids);
    auto [slot, plen] = walk(b.ptr, b.n);
    (void)plen;
    // Walk up to the deepest node whose END boundary owns a LIVE snapshot.
    int32_t cur = slot;
    int64_t end_len = path_len(slot);
    while (cur != root_) {
      Node &cn = nodes_[cur];
      if (cn.mamba_value != -1) {
        return py::make_tuple(path_kv(cur), end_len, (int64_t)cn.mamba_value,
                              ref(make_id(cur, cn.gen)));
      }
      end_len -= (int64_t)cn.key.size();
      cur = cn.parent;
    }
    return py::make_tuple(empty_, (int64_t)0, py::none(), ref(root_id()));
  }

  py::tuple insert_hybrid(const torch::Tensor &ids, const torch::Tensor &kv,
                          int64_t mamba_slot) {
    TORCH_CHECK(mamba_slot >= 0, "hybrid insert requires a snapshot slot");
    IdsBuf b = ids_buf(ids);
    int64_t insert_len = align_down_i(b.n, page_);
    auto [slot, plen] = walk(b.ptr, insert_len);
    if (plen != insert_len) {
      int32_t ns = new_node(tic());
      Node &nn = nodes_[ns];
      set_key_value(nn, b.ptr + plen, insert_len - plen,
                    kv.narrow(0, plen, insert_len - plen).clone());
      nn.parent = slot;
      register_child(slot, ns);
      evictable_ += (int64_t)nn.key.size();
      slot = ns;
    }
    if (slot == root_) {
      // root can't hold a snapshot; report exist so the caller frees its slot
      return py::make_tuple(plen, true);
    }
    Node &n = nodes_[slot];
    if (n.mamba_value != -1) {
      return py::make_tuple(plen, true);  // dedup: caller frees the donated slot
    }
    n.mamba_value = (int32_t)mamba_slot;  // fills a fresh node or a tombstone
    if (n.mamba_ref == 0) mamba_evictable_ += 1;
    return py::make_tuple(plen, false);
  }

  void inc_lock_hybrid(int64_t id) {
    node(id);
    Node &n = nodes_[slot_of(id)];
    if (n.mamba_value != -1) {
      if (n.mamba_ref == 0) {
        mamba_evictable_ -= 1;
        mamba_protected_ += 1;
      }
      n.mamba_ref += 1;
    }
    full_chain_lock(slot_of(id), false);
  }

  void dec_lock_hybrid(int64_t id) {
    node(id);
    Node &n = nodes_[slot_of(id)];
    if (n.mamba_value != -1 && n.mamba_ref > 0) {
      n.mamba_ref -= 1;
      if (n.mamba_ref == 0) {
        mamba_evictable_ += 1;
        mamba_protected_ -= 1;
      }
    }
    full_chain_lock(slot_of(id), true);
  }

  py::tuple evict_full_hybrid(int64_t num_tokens) {
    EvictHeap heap;
    collect_unlocked_leaves(heap);
    std::vector<torch::Tensor> kv;
    std::vector<int64_t> mamba;
    std::vector<int32_t> unlinked;
    int64_t freed = 0;
    while (freed < num_tokens && !heap.empty()) {
      auto [stamp, slot] = heap.top();
      heap.pop();
      (void)stamp;
      Node &n = nodes_[slot];
      if (n.dead || n.ref_count != 0 || !is_leaf(slot) || slot == root_) continue;
      freed += (int64_t)n.key.size();
      kv.push_back(n.value);
      evictable_ -= (int64_t)n.key.size();
      free_node_mamba(n, mamba);
      int32_t parent = n.parent;
      unlink(slot, unlinked);
      auto [survivor, casc] = cascade_mamba_tombstone_leaves(parent, kv, unlinked);
      freed += casc;
      if (is_leaf(survivor) && nodes_[survivor].ref_count == 0 && survivor != root_) {
        heap.emplace(nodes_[survivor].stamp, survivor);
      }
    }
    finish_unlinked(unlinked);
    return py::make_tuple(cat_or_empty(kv), mamba);
  }

  py::tuple evict_mamba(int64_t num) {
    EvictHeap heap;
    for (int32_t s = 1; s < (int32_t)nodes_.size(); ++s) {
      Node &n = nodes_[s];
      if (!n.dead && n.mamba_value != -1 && n.mamba_ref == 0) heap.emplace(n.stamp, s);
    }
    std::vector<torch::Tensor> kv;
    std::vector<int64_t> mamba;
    std::vector<int32_t> unlinked;
    int64_t freed = 0;
    while (freed < num && !heap.empty()) {
      auto [stamp, slot] = heap.top();
      heap.pop();
      (void)stamp;
      Node &n = nodes_[slot];
      if (n.dead || n.mamba_value == -1 || n.mamba_ref != 0 || slot == root_) continue;
      if (is_leaf(slot) && n.ref_count == 0) {
        kv.push_back(n.value);
        evictable_ -= (int64_t)n.key.size();
        free_node_mamba(n, mamba);
        freed += 1;
        int32_t parent = n.parent;
        unlink(slot, unlinked);
        cascade_mamba_tombstone_leaves(parent, kv, unlinked);
      } else {
        free_node_mamba(n, mamba);  // tombstone internal (or locked-KV) node
        freed += 1;
      }
    }
    finish_unlinked(unlinked);
    return py::make_tuple(cat_or_empty(kv), mamba);
  }

  std::string check_integrity_hybrid() {
    for (int32_t s = 1; s < (int32_t)nodes_.size(); ++s) {
      Node &n = nodes_[s];
      if (n.dead) continue;
      if (n.mamba_value != -1 && (n.mamba_ref < 0 || n.ref_count < 0)) {
        return "snapshot node with negative refs";
      }
      if (n.ref_count < 0) return "negative full ref count";
    }
    return "";
  }

  // ==================================================================
  // SWA (SWARadixCache)
  // ==================================================================
  py::tuple match_swa(const torch::Tensor &ids) {
    IdsBuf b = ids_buf(ids);
    const int64_t *ptr = b.ptr;
    int64_t total = b.n;
    int32_t slot = root_;
    std::vector<torch::Tensor> value;
    // A path connected to root without a tombstone is always reusable: the run
    // starts "infinite" (saturating: never incremented past kRunInf).
    constexpr int64_t kRunInf = INT64_MAX / 4;
    int64_t match_since_tomb = kRunInf;
    int64_t best_value_len = 0;
    int32_t best_slot = root_;
    int64_t pos = 0;
    while (pos < total) {
      const int64_t *q = ptr + pos;
      int64_t qlen = total - pos;
      int32_t cid = lookup_child(slot, q, qlen);
      if (cid < 0) break;
      if (nodes_[cid].swa_tomb) {
        // Commit the windowed-safe boundary at the PARENT (before this tombstone)
        // if the live run leading up to it already covers the window; reset it.
        if (match_since_tomb >= window_) {
          best_value_len = (int64_t)value.size();
          best_slot = slot;
        }
        match_since_tomb = 0;
      }
      int64_t ml = align_down_i(mismatch_len(nodes_[cid].key, q, qlen), page_);
      if (ml == 0) break;
      if (ml < (int64_t)nodes_[cid].key.size()) {
        int32_t pslot = split_at(cid, ml);  // prefix half := the matched segment
        value.push_back(nodes_[pslot].value);
        if (!nodes_[pslot].swa_tomb) {
          if (match_since_tomb < kRunInf) match_since_tomb += (int64_t)nodes_[pslot].key.size();
        }
        slot = pslot;
        pos += ml;
        break;
      }
      value.push_back(nodes_[cid].value);
      if (!nodes_[cid].swa_tomb) {
        if (match_since_tomb < kRunInf) match_since_tomb += (int64_t)nodes_[cid].key.size();
      }
      slot = cid;
      pos += ml;
    }
    if (match_since_tomb >= window_) {
      best_value_len = (int64_t)value.size();
      best_slot = slot;
    }
    stamp_path(best_slot);
    std::vector<torch::Tensor> best(value.begin(), value.begin() + best_value_len);
    torch::Tensor kv = cat_or_empty_copy(best);
    Node &bn = nodes_[best_slot];
    return py::make_tuple(kv, (int64_t)kv.numel(), ref(make_id(best_slot, bn.gen)));
  }

  py::tuple insert_swa(const torch::Tensor &ids, const torch::Tensor &kv_indices,
                       int64_t swa_evicted_seqlen, int64_t update_kv_after_len) {
    IdsBuf b = ids_buf(ids);
    int64_t insert_len = align_down_i(b.n, page_);
    const int64_t *ptr = b.ptr;
    std::vector<torch::Tensor> freed;
    int32_t slot = root_;
    int64_t total = 0;
    while (total < insert_len) {
      const int64_t *q = ptr + total;
      int64_t qlen = insert_len - total;
      int32_t cid = lookup_child(slot, q, qlen);
      if (cid < 0) break;
      int64_t ml = align_down_i(mismatch_len(nodes_[cid].key, q, qlen), page_);
      if (ml == 0) break;
      bool partial = ml < (int64_t)nodes_[cid].key.size();
      if (partial) {
        cid = split_at(cid, ml);  // cid := matched prefix half (orig stays suffix)
      }
      Node &child = nodes_[cid];
      torch::Tensor seg = kv_indices.narrow(0, total, ml);
      if (update_kv_after_len < total + ml) {
        if (child.swa_tomb) {
          TORCH_CHECK(child.swa_ref == 0, "a tombstoned node cannot hold a swa lock");
          if (child.ref_count > 0) {
            // A full-locked reader still gathers the node's CURRENT slots: keep
            // the tombstone and the tree's value, drop the dup (Branch 3 shape).
            freed.push_back(seg.clone());
          } else if (swa_evicted_seqlen <= total) {
            // Branch 1: the node's swa is live in the request -> revive it whole.
            freed.push_back(child.value);
            child.value = seg.clone();
            child.swa_tomb = false;
            child.stamp = stride_tic();
            swa_evictable_ += (int64_t)child.key.size();
            revives_ += 1;
          } else if (swa_evicted_seqlen < total + ml) {
            // Branch 2: the request's freed-swa frontier falls inside the node.
            // Split; the head stays tombstone, revive the live tail. (Identity:
            // split_at returns the head slot; `cid` remains the tail.)
            int64_t start = swa_evicted_seqlen - total;
            split_at(cid, start);
            Node &tail = nodes_[cid];
            freed.push_back(tail.value);                       // tail's old slots
            freed.push_back(seg.narrow(0, 0, start).clone());  // dup for the head
            tail.value = seg.narrow(0, start, ml - start).clone();
            tail.swa_tomb = false;
            tail.stamp = stride_tic();
            swa_evictable_ += (int64_t)tail.key.size();
            revives_ += 1;
          } else {
            // Branch 3: still wholly out-of-window: keep the tombstone, drop the dup.
            freed.push_back(seg.clone());
          }
        } else {
          // Matched a live node: the tree slots are canonical; drop the dup.
          freed.push_back(seg.clone());
        }
      }
      total += ml;
      slot = cid;
      if (partial) break;
    }
    // Suffix: tombstone [total, swa_evicted_seqlen), then a LIVE leaf for the
    // in-window remainder (a leaf is never a tombstone -- the clamp plus the
    // free_swa -page margin guarantee a live tail). NOTE: `total` stays the
    // MATCHED prefix length (the return value); the suffix grows `off` only.
    if (total < insert_len) {
      int64_t boundary = std::max<int64_t>(0, std::min(swa_evicted_seqlen, insert_len) - total);
      boundary = std::min(boundary, std::max<int64_t>(0, insert_len - total - page_));
      int64_t off = total;
      if (boundary > 0) {
        slot = add_child_swa(slot, ptr + off, boundary,
                             kv_indices.narrow(0, off, boundary), true);
        off += boundary;
      }
      if (off < insert_len) {
        add_child_swa(slot, ptr + off, insert_len - off,
                      kv_indices.narrow(0, off, insert_len - off), false);
      }
    }
    return py::make_tuple(total, cat_or_empty(freed));
  }

  int32_t add_child_swa(int32_t parent, const int64_t *ids, int64_t n,
                        const torch::Tensor &kv, bool tomb) {
    int32_t s = new_node(stride_tic());
    Node &c = nodes_[s];
    set_key_value(c, ids, n, kv.clone());
    c.parent = parent;
    register_child(parent, s);
    c.swa_tomb = tomb;
    evictable_ += n;
    if (!tomb) swa_evictable_ += n;
    return s;
  }

  py::object inc_lock_swa(int64_t id) {
    node(id);
    int32_t s = slot_of(id);
    py::object uuid_out = py::none();
    int64_t swa_locked = 0;
    while (s != root_) {
      Node &cur = nodes_[s];
      if (cur.ref_count == 0) {
        evictable_ -= (int64_t)cur.key.size();
        protected_ += (int64_t)cur.key.size();
      }
      cur.ref_count += 1;
      if (swa_locked < window_ && !cur.swa_tomb) {
        if (cur.swa_ref == 0) {
          swa_evictable_ -= (int64_t)cur.key.size();
          swa_protected_ += (int64_t)cur.key.size();
        }
        cur.swa_ref += 1;
        swa_locked += (int64_t)cur.key.size();
        if (swa_locked >= window_) {
          if (cur.swa_uuid == -1) cur.swa_uuid = ++swa_uuid_ctr_;
          uuid_out = py::int_(cur.swa_uuid);
        }
      }
      s = cur.parent;
    }
    return uuid_out;
  }

  void dec_lock_swa(int64_t id, py::object uuid_obj, bool skip_swa) {
    node(id);
    int32_t s = slot_of(id);
    bool dec_swa = !skip_swa;
    int64_t uuid = uuid_obj.is_none() ? -1 : uuid_obj.cast<int64_t>();
    while (s != root_) {
      Node &cur = nodes_[s];
      cur.ref_count -= 1;
      TORCH_CHECK(cur.ref_count >= 0, "swa radix: ref_count underflow on unlock");
      if (cur.ref_count == 0) {
        evictable_ += (int64_t)cur.key.size();
        protected_ -= (int64_t)cur.key.size();
      }
      if (dec_swa && !cur.swa_tomb && cur.swa_ref > 0) {
        cur.swa_ref -= 1;
        if (cur.swa_ref == 0) {
          swa_evictable_ += (int64_t)cur.key.size();
          swa_protected_ -= (int64_t)cur.key.size();
        }
        if (uuid != -1 && cur.swa_uuid == uuid) {
          dec_swa = false;  // window boundary reached (inclusive)
        }
      }
      s = cur.parent;
    }
  }

  py::tuple evict_full_swa(int64_t num_tokens) {
    EvictHeap heap;
    collect_unlocked_leaves(heap);
    std::vector<torch::Tensor> kv, swa;
    std::vector<int32_t> unlinked;
    int64_t freed = 0;
    while (freed < num_tokens && !heap.empty()) {
      auto [stamp, slot] = heap.top();
      heap.pop();
      (void)stamp;
      Node &n = nodes_[slot];
      if (n.dead || n.ref_count != 0 || !is_leaf(slot) || slot == root_) continue;
      freed += (int64_t)n.key.size();
      kv.push_back(n.value);
      evictable_ -= (int64_t)n.key.size();
      if (!n.swa_tomb) {
        swa.push_back(n.value);  // its swa is still live -> free it too
        swa_evictable_ -= (int64_t)n.key.size();
      }
      int32_t parent = n.parent;
      unlink(slot, unlinked);
      // Re-push the SURVIVING ancestor the cascade returned (the original parent
      // may itself have been unlinked -- re-pushing it would double-free).
      auto [survivor, casc] = cascade_swa_tombstone_leaves(parent, kv, unlinked);
      freed += casc;
      if (is_leaf(survivor) && nodes_[survivor].ref_count == 0 && survivor != root_) {
        heap.emplace(nodes_[survivor].stamp, survivor);
      }
    }
    finish_unlinked(unlinked);
    return py::make_tuple(cat_or_empty(kv), cat_or_empty(swa));
  }

  py::tuple evict_swa(int64_t num_tokens) {
    EvictHeap heap;
    for (int32_t s = 1; s < (int32_t)nodes_.size(); ++s) {
      Node &n = nodes_[s];
      if (!n.dead && !n.swa_tomb && n.swa_ref == 0) heap.emplace(n.stamp, s);
    }
    std::vector<torch::Tensor> kv, swa;
    std::vector<int32_t> unlinked;
    int64_t freed = 0;
    while (freed < num_tokens && !heap.empty()) {
      auto [stamp, slot] = heap.top();
      heap.pop();
      (void)stamp;
      Node &n = nodes_[slot];
      if (n.dead || n.swa_tomb || n.swa_ref != 0 || slot == root_) continue;
      if (is_leaf(slot) && n.ref_count == 0) {
        kv.push_back(n.value);
        swa.push_back(n.value);
        evictable_ -= (int64_t)n.key.size();
        swa_evictable_ -= (int64_t)n.key.size();
        freed += (int64_t)n.key.size();
        n.swa_tomb = true;
        int32_t parent = n.parent;
        unlink(slot, unlinked);
        cascade_swa_tombstone_leaves(parent, kv, unlinked);
      } else {
        swa.push_back(n.value);  // tombstone internal / full-locked leaf in place
        swa_evictable_ -= (int64_t)n.key.size();
        freed += (int64_t)n.key.size();
        n.swa_tomb = true;
      }
    }
    finish_unlinked(unlinked);
    return py::make_tuple(cat_or_empty(kv), cat_or_empty(swa));
  }

  torch::Tensor trim_head_swa(const torch::Tensor &ids, int64_t keep_from) {
    if (keep_from <= 0) return empty_;
    // Ensure a node boundary at keep_from (the Python re-match splits + stamps).
    match_swa(ids.narrow(0, 0, std::min<int64_t>(keep_from, ids.numel())));
    IdsBuf b = ids_buf(ids);
    std::vector<torch::Tensor> freed;
    int32_t slot = root_;
    int64_t pos = 0;
    while (pos < keep_from) {
      int32_t cid = lookup_child(slot, b.ptr + pos, std::min(keep_from - pos, b.n - pos));
      if (cid < 0) break;
      Node &c = nodes_[cid];
      if (pos + (int64_t)c.key.size() > keep_from) break;
      if (!c.swa_tomb && c.swa_ref == 0 && !is_leaf(cid)) {
        freed.push_back(c.value);
        swa_evictable_ -= (int64_t)c.key.size();
        c.swa_tomb = true;
      }
      slot = cid;
      pos += (int64_t)c.key.size();
    }
    return cat_or_empty(freed);
  }

  std::string check_integrity_swa() {
    for (int32_t s = 1; s < (int32_t)nodes_.size(); ++s) {
      Node &n = nodes_[s];
      if (n.dead) continue;
      if (n.ref_count < 0 || n.swa_ref < 0) return "negative ref counts";
      if (n.ref_count < n.swa_ref) return "full_ref must be >= swa_ref";
      if (n.swa_tomb && n.swa_ref != 0) return "a tombstoned node cannot hold a swa lock";
    }
    return "";
  }

  // ==================================================================
  // shared helpers
  // ==================================================================
  void full_chain_lock(int32_t start, bool unlock) {
    int32_t s = start;
    if (unlock) {
      while (s != root_) {
        Node &cur = nodes_[s];
        cur.ref_count -= 1;
        TORCH_CHECK(cur.ref_count >= 0, "radix: ref_count underflow on unlock");
        if (cur.ref_count == 0) {
          evictable_ += (int64_t)cur.key.size();
          protected_ -= (int64_t)cur.key.size();
        }
        s = cur.parent;
      }
    } else {
      while (s != root_) {
        Node &cur = nodes_[s];
        if (cur.ref_count == 0) {
          evictable_ -= (int64_t)cur.key.size();
          protected_ += (int64_t)cur.key.size();
        }
        cur.ref_count += 1;
        s = cur.parent;
      }
    }
  }

  void free_node_mamba(Node &n, std::vector<int64_t> &out) {
    if (n.mamba_value != -1) {
      out.push_back(n.mamba_value);
      n.mamba_value = -1;
      if (n.mamba_ref == 0) mamba_evictable_ -= 1;
    }
  }

  // After a leaf unlink, eagerly reclaim the exposed KV-only tombstone leaves
  // upward (hybrid: mamba_value None -- a leaf always carries a live snapshot).
  // Returns (highest surviving ancestor, freed tokens).
  std::pair<int32_t, int64_t> cascade_mamba_tombstone_leaves(
      int32_t parent, std::vector<torch::Tensor> &kv, std::vector<int32_t> &unlinked) {
    int64_t freed = 0;
    int32_t p = parent;
    while (p != root_ && !nodes_[p].dead && nodes_[p].mamba_value == -1 && is_leaf(p) &&
           nodes_[p].ref_count == 0) {
      Node &pn = nodes_[p];
      kv.push_back(pn.value);
      evictable_ -= (int64_t)pn.key.size();
      freed += (int64_t)pn.key.size();
      int32_t next = pn.parent;
      unlink(p, unlinked);
      p = next;
    }
    return {p, freed};
  }

  // SWA sibling: cascade over exposed swa-tombstone leaves (their swa currency
  // was already freed when they were tombstoned, so only full KV is reclaimed).
  std::pair<int32_t, int64_t> cascade_swa_tombstone_leaves(
      int32_t parent, std::vector<torch::Tensor> &kv, std::vector<int32_t> &unlinked) {
    int64_t freed = 0;
    int32_t p = parent;
    while (p != root_ && !nodes_[p].dead && nodes_[p].swa_tomb && is_leaf(p) &&
           nodes_[p].ref_count == 0) {
      Node &pn = nodes_[p];
      kv.push_back(pn.value);
      evictable_ -= (int64_t)pn.key.size();
      freed += (int64_t)pn.key.size();
      int32_t next = pn.parent;
      unlink(p, unlinked);
      p = next;
    }
    return {p, freed};
  }

  void unlink(int32_t slot, std::vector<int32_t> &unlinked) {
    Node &n = nodes_[slot];
    TORCH_CHECK(is_leaf(slot), "radix: refusing to unlink a non-leaf");
    // The key is still intact here (only split mutates keys, and it rewrites the
    // bucket itself), so the first-page hash finds the bucket in O(1).
    Node &p = nodes_[n.parent];
    uint64_t h = node_hash(n);
    auto it = p.children.find(h);
    TORCH_CHECK(it != p.children.end(), "radix: unlinked node missing from its bucket");
    auto &v = it->second;
    auto at = std::find(v.begin(), v.end(), slot);
    TORCH_CHECK(at != v.end(), "radix: unlinked node missing from its bucket");
    v.erase(at);
    if (v.empty()) p.children.erase(it);
    n.dead = true;
    n.key.clear();
    n.key.shrink_to_fit();
    n.value = torch::Tensor();  // drop the device buffer
    unlinked.push_back(slot);
  }

  void finish_unlinked(const std::vector<int32_t> &unlinked) {
    for (int32_t s : unlinked) {
      prune_ref(make_id(s, nodes_[s].gen));
      free_slots_.push_back(s);
    }
  }

  void collect_unlocked_leaves(EvictHeap &heap) {
    for (int32_t s = 1; s < (int32_t)nodes_.size(); ++s) {
      Node &n = nodes_[s];
      if (!n.dead && n.ref_count == 0 && is_leaf(s)) heap.emplace(n.stamp, s);
    }
  }

  void stamp_path(int32_t slot) {
    // Strictly DECREASING toward the root (deepest newest): the eviction heap
    // reclaims near-root swa nodes first (sglang reset_node_and_parents_mru).
    int64_t base = stride_tic();
    int64_t off = 0;
    int32_t s = slot;
    while (s != root_) {
      nodes_[s].stamp = base - off;
      off += 1;
      s = nodes_[s].parent;
    }
  }

  torch::Tensor cat_or_empty(const std::vector<torch::Tensor> &ts) {
    if (ts.empty()) return empty_;
    if (ts.size() == 1) return ts[0];
    return torch::cat(ts);
  }

  // Match results alias-copy like path_kv: the Python torch.cat copies, so a
  // single-node match must not hand out the live node's tensor either.
  torch::Tensor cat_or_empty_copy(const std::vector<torch::Tensor> &ts) {
    if (ts.empty()) return empty_;
    if (ts.size() == 1) return ts[0].clone();
    return torch::cat(ts);
  }

  // ---- input tensor -> int64 host view -------------------------------------
  // ids arrive as 1-D CPU int32 (production token pool) or int64 (tests). The
  // int64 path borrows the buffer; int32 widens into a scratch vector that lives
  // until the next ids_buf call on this tree (one walk at a time, single thread).
  struct IdsBuf {
    const int64_t *ptr = nullptr;
    int64_t n = 0;
  };
  IdsBuf ids_buf(const torch::Tensor &ids) {
    TORCH_CHECK(ids.dim() == 1 && ids.is_cpu(), "radix: ids must be a 1-D CPU tensor");
    TORCH_CHECK(ids.dtype() == torch::kInt64 || ids.dtype() == torch::kInt32,
                "radix: ids must be int32 or int64");
    TORCH_CHECK(ids.is_contiguous(), "radix: ids must be contiguous");
    IdsBuf b;
    b.n = ids.numel();
    if (ids.dtype() == torch::kInt64) {
      b.ptr = b.n ? ids.data_ptr<int64_t>() : nullptr;
    } else {
      ids_scratch_.resize(b.n);
      const int32_t *src = ids.data_ptr<int32_t>();
      for (int64_t i = 0; i < b.n; ++i) ids_scratch_[i] = src[i];
      b.ptr = b.n ? ids_scratch_.data() : nullptr;
    }
    return b;
  }

  // ---- introspection (tests / integrity) ------------------------------------
  int64_t node_length(int64_t id) { return (int64_t)node(id).key.size(); }
  torch::Tensor node_key(int64_t id) {
    Node &n = node(id);
    auto t = torch::empty({(int64_t)n.key.size()},
                          torch::TensorOptions().dtype(torch::kInt64));
    if (!n.key.empty()) memcpy(t.data_ptr<int64_t>(), n.key.data(), n.key.size() * 8);
    return t;
  }
  torch::Tensor node_value(int64_t id) {
    torch::Tensor v = node(id).value;
    TORCH_CHECK(v.defined(), "radix: node has no value");
    return v;
  }
  int64_t node_parent(int64_t id) {
    Node &n = node(id);
    return n.parent < 0 ? -1 : make_id((int32_t)n.parent, nodes_[n.parent].gen);
  }
  py::list node_children(int64_t id) {
    Node &n = node(id);
    py::list out;
    for (auto &bucket : n.children) {
      for (int32_t cid : bucket.second) out.append(ref(make_id(cid, nodes_[cid].gen)));
    }
    return out;
  }
  int64_t node_stamp(int64_t id) { return node(id).stamp; }
  std::vector<int64_t> node_fields(int64_t id) {
    Node &n = node(id);
    return {n.ref_count, n.mamba_value, n.mamba_ref, (int64_t)n.swa_tomb, n.swa_ref,
            n.swa_uuid, n.uuid};
  }
  bool node_is_root(int64_t id) { return node(id).parent < 0; }
  bool node_is_leaf(int64_t id) {
    node(id);
    return is_leaf(slot_of(id));
  }
  py::object node_split(int64_t id, int64_t pos) {
    node(id);
    int32_t ps = split_at(slot_of(id), pos);
    return ref(make_id(ps, nodes_[ps].gen));
  }
  int64_t node_match_len(int64_t id, const torch::Tensor &q) {
    Node &n = node(id);
    IdsBuf b = ids_buf(q);
    return mismatch_len(n.key, b.ptr, b.n);
  }
  py::list path_slots(int64_t id) {
    node(id);
    torch::Tensor kv = path_kv(slot_of(id));
    py::list out;
    if (kv.numel() == 0) return out;
    auto cpu = kv.is_cpu() ? kv : kv.to(torch::kCPU);
    if (cpu.dtype() == torch::kInt32) {
      const int32_t *p = cpu.data_ptr<int32_t>();
      for (int64_t i = 0; i < cpu.numel(); ++i) out.append((int64_t)p[i]);
    } else {
      const int64_t *p = cpu.data_ptr<int64_t>();
      for (int64_t i = 0; i < cpu.numel(); ++i) out.append(p[i]);
    }
    return out;
  }
  std::string check_integrity() {
    if (mode_ == Mode::Hybrid) return check_integrity_hybrid();
    if (mode_ == Mode::SWA) return check_integrity_swa();
    return "";
  }
  int64_t mode() const { return (int64_t)mode_; }
  int64_t page_size() const { return page_; }
  int64_t num_nodes() {
    int64_t c = 0;
    for (int32_t s = 1; s < (int32_t)nodes_.size(); ++s)
      if (!nodes_[s].dead) ++c;
    return c;
  }
  py::object root_ref() { return ref(root_id()); }

 private:
  Mode mode_;
  int64_t page_;
  int64_t window_;
  torch::TensorOptions opts_;
  torch::Tensor empty_;
  std::deque<Node> nodes_;              // deque: element refs survive growth
  std::vector<uint32_t> gen_ctr_;       // per-slot generation (parallel to nodes_)
  std::vector<int32_t> free_slots_;
  int32_t root_ = 0;
  int64_t clk_ = 0;
  int64_t uuid_ctr_ = 0;
  int64_t swa_uuid_ctr_ = 0;
  std::unordered_map<int64_t, py::object> refs_;  // strong NodeRef registry (live ids)
  std::vector<int64_t> ids_scratch_;              // int32->int64 widening scratch
};

std::shared_ptr<TreeCore> NodeRef::tree() const {
  auto t = tree_.lock();
  TORCH_CHECK(t, "radix tree was destroyed");
  return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<NodeRef>(m, "NodeRef")
      .def("__repr__",
           [](const NodeRef &n) { return "<RadixNode id=" + std::to_string(n.id()) + ">"; })
      .def("__hash__", [](const NodeRef &n) { return std::hash<int64_t>{}(n.id()); })
      .def("__eq__",
           [](const NodeRef &a, py::handle b) {
             if (!py::isinstance<NodeRef>(b)) return false;
             return a.id() == b.cast<NodeRef>().id();
           })
      .def_property_readonly("id64", &NodeRef::id)
      .def_property_readonly("length",
                             [](const NodeRef &n) { return n.tree()->node_length(n.id()); })
      .def_property_readonly("_key",
                             [](const NodeRef &n) { return n.tree()->node_key(n.id()); })
      .def_property_readonly("value",
                             [](const NodeRef &n) { return n.tree()->node_value(n.id()); })
      .def_property_readonly("ref_count", [](const NodeRef &n) {
        return n.tree()->node_fields(n.id())[0];
      })
      .def_property_readonly("timestamp",
                             [](const NodeRef &n) { return n.tree()->node_stamp(n.id()); })
      .def_property_readonly("mamba_value", [](const NodeRef &n) -> py::object {
        int64_t v = n.tree()->node_fields(n.id())[1];
        return v < 0 ? py::object(py::none()) : py::object(py::int_(v));
      })
      .def_property_readonly("mamba_ref_count", [](const NodeRef &n) {
        return n.tree()->node_fields(n.id())[2];
      })
      .def_property_readonly("swa_tombstone", [](const NodeRef &n) {
        return n.tree()->node_fields(n.id())[3] != 0;
      })
      .def_property_readonly("swa_ref_count",
                             [](const NodeRef &n) { return n.tree()->node_fields(n.id())[4]; })
      .def_property_readonly("swa_uuid", [](const NodeRef &n) -> py::object {
        int64_t v = n.tree()->node_fields(n.id())[5];
        return v < 0 ? py::object(py::none()) : py::object(py::int_(v));
      })
      .def_property_readonly("uuid",
                             [](const NodeRef &n) { return n.tree()->node_fields(n.id())[6]; })
      .def_property_readonly("_parent", [](const NodeRef &n) -> py::object {
        int64_t p = n.tree()->node_parent(n.id());
        return p < 0 ? py::none() : n.tree()->ref(p);
      })
      .def_property_readonly("parent", [](const NodeRef &n) -> py::object {
        int64_t p = n.tree()->node_parent(n.id());
        TORCH_CHECK(p >= 0, "radix: root has no parent");
        return n.tree()->ref(p);
      })
      .def_property_readonly("children", [](const NodeRef &n) -> py::dict {
        auto t = n.tree();
        // Keyed exactly like the Python key_fn output (scalar at page_size 1,
        // tuple of the first page's tokens otherwise) so the battery's
        // `parent.children.get(cache.key_fn(node._key)) is node` holds.
        py::list flat = t->node_children(n.id());
        int64_t P = t->page_size();
        py::dict d;
        for (auto h : flat) {
          const NodeRef &nr = h.cast<const NodeRef &>();
          torch::Tensor key = t->node_key(nr.id());
          const int64_t *kp = key.data_ptr<int64_t>();
          int64_t kn = std::min(P, key.numel());
          py::object pykey;
          if (P == 1) {
            pykey = py::int_(kp[0]);
          } else {
            py::tuple tup(kn);
            for (int64_t i = 0; i < kn; ++i) tup[i] = py::int_(kp[i]);
            pykey = std::move(tup);
          }
          d[pykey] = h;
        }
        return d;
      })
      .def("is_root", [](const NodeRef &n) { return n.tree()->node_is_root(n.id()); })
      .def("is_leaf", [](const NodeRef &n) { return n.tree()->node_is_leaf(n.id()); })
      .def("split_at",
           [](const NodeRef &n, int64_t pos) { return n.tree()->node_split(n.id(), pos); })
      .def("get_match_len", [](const NodeRef &n, const torch::Tensor &q) {
        return n.tree()->node_match_len(n.id(), q);
      })
      .def("path_kv", [](const NodeRef &n) { return n.tree()->path_kv_id(n.id()); })
      .def("path_slots", [](const NodeRef &n) { return n.tree()->path_slots(n.id()); });

  py::class_<TreeCore, std::shared_ptr<TreeCore>>(m, "RadixTree")
      .def(py::init<int, int64_t, int64_t, std::string>(), py::arg("mode"),
           py::arg("page_size"), py::arg("window") = 0, py::arg("device") = "cpu")
      // identity / introspection
      .def("root_ref", &TreeCore::root_ref)
      .def("ref", &TreeCore::ref)
      .def("counters", &TreeCore::counters)
      .def("mode", &TreeCore::mode)
      .def("page_size", &TreeCore::page_size)
      .def("num_nodes", &TreeCore::num_nodes)
      .def("path_kv", &TreeCore::path_kv_id)
      .def("path_slots", &TreeCore::path_slots)
      .def("check_integrity", &TreeCore::check_integrity)
      // plain
      .def("match_plain", &TreeCore::match_plain)
      .def("insert_plain", &TreeCore::insert_plain)
      .def("lock_plain", &TreeCore::lock_plain)
      .def("evict_plain", &TreeCore::evict_plain)
      // hybrid
      .def("match_hybrid", &TreeCore::match_hybrid)
      .def("insert_hybrid", &TreeCore::insert_hybrid)
      .def("inc_lock_hybrid", &TreeCore::inc_lock_hybrid)
      .def("dec_lock_hybrid", &TreeCore::dec_lock_hybrid)
      .def("evict_full_hybrid", &TreeCore::evict_full_hybrid)
      .def("evict_mamba", &TreeCore::evict_mamba)
      // swa
      .def("match_swa", &TreeCore::match_swa)
      .def("insert_swa", &TreeCore::insert_swa)
      .def("inc_lock_swa", &TreeCore::inc_lock_swa)
      .def("dec_lock_swa", &TreeCore::dec_lock_swa)
      .def("evict_full_swa", &TreeCore::evict_full_swa)
      .def("evict_swa", &TreeCore::evict_swa)
      .def("trim_head_swa", &TreeCore::trim_head_swa);
}
