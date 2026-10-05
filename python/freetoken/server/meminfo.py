"""VRAM breakdown for GET /v1/meminfo (JSON) and GET /meminfo (HTML dashboard).

Every figure is computed from the readiness ("meta", ...) ack that EVERY TP worker pushes for
its own rank (device identity, weights, pool allocation, per-unit costs) plus the live gauges
on the stats tracker -- the same measured numbers the desktop cache panel prices with, and
nothing here touches the scheduler or CUDA. TP figures are SUMMED across the per-rank acks
(fleet totals); the live gauges are rank0-sampled only and labeled as such where they surface.

Pool semantics follow the Task Manager reading: a pool's headline number is what it actually
holds (active-kept content + evictable warm cache), NOT just what one request keeps pinned --
otherwise an idle engine would show its history as 0 while the pages sit right there in the
pool. Residual is derived on rank0 only (the rank whose live VRAM is measured); the
unmeasured remainder of the other ranks folds into free (est.).
"""

from __future__ import annotations

import time
from typing import Any

_GIB = 1024**3

# (key, label, unit-count key, page-count key, unit name, per-unit cost key). Page-keyed
# pools price in tokens (pages x page_size); slot pools in slots.
_POOL_DEFS = (
    ("kv", "KV プール", "num_pages", "page_size", "tokens", "kv_bytes_per_token"),
    ("swa", "SWA プール", "num_swa_pages", "swa_page_size", "tokens", "swa_bytes_per_token"),
    ("moe", "MoE エキスパートプール", "moe_cache_size", None, "スロット", "moe_bytes_per_expert"),
    ("mamba", "GDN 状態プール", "num_mamba_slots", None, "スロット", "mamba_bytes_per_slot"),
)


def _i(obj: Any, key: str, default: int = 0) -> int:
    try:
        return int(getattr(obj, key, default) if not isinstance(obj, dict) else obj.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _rank_payloads(state: Any) -> list:
    """[(rank, raw meta)] sorted, from the per-rank readiness acks. Falls back to the old
    flat single-rank fields (older backend, or still loading) as one rank-0 payload."""
    metas = getattr(state, "rank_metas", None) or {}
    if metas:
        try:
            return sorted(((int(k), v or {}) for k, v in metas.items()), key=lambda t: t[0])
        except (TypeError, ValueError):
            pass
    ub = getattr(state, "unit_bytes", None) or {}
    flat = {
        "weights_bytes": _i(state, "weights_bytes"),
        "device_total_bytes": _i(state, "device_total_bytes"),
        "nonpool_overhead_bytes": _i(state, "nonpool_overhead_bytes"),
        "cache_budget_bytes": _i(state, "cache_budget_bytes"),
        "free_vram_bytes": _i(state, "free_vram_bytes"),
        "gpus": list(getattr(state, "gpus", None) or []),
        "pools": getattr(state, "cache_pools", None) or {},
    }
    for key in ("kv_bytes_per_token", "moe_bytes_per_expert", "mamba_bytes_per_slot",
                "swa_bytes_per_token"):
        flat[key] = _i(ub, key)
    return [(0, flat)]


def _pool_alloc(pools: dict, ub: dict, cfg: Any, key: str) -> tuple:
    """(logical units, per-rank unit cost, bytes) of one pool on one rank; 0s when unknown."""
    for k, _label, unit_key, page_key, _unit, cost_key in _POOL_DEFS:
        if k != key:
            continue
        units = _i(pools, unit_key)
        if page_key and units > 0:
            fallback = _i(cfg, "page_size", 1) if page_key == "page_size" else 1
            units *= max(1, _i(pools, page_key, fallback))
        cost = _i(ub, cost_key)
        return units, cost, units * cost
    return 0, 0, 0


def _moe_ratios(stats: Any) -> tuple:
    """(working-set, warm-cache) fill fractions of the MoE slot cache, or (None, None) when
    the gauge never ran. Both are fractions of the cache: `active` is what the last forward
    read, the rest of the fill is warm cache, and nothing in this pool is kept from eviction."""
    total = _i(stats, "moe_total_slots")
    if total <= 0:
        return None, None
    filled = min(_i(stats, "moe_used_slots"), total)
    active = min(_i(stats, "moe_active_slots"), filled)
    return active / total, (filled - active) / total


def build_meminfo(state: Any) -> dict:
    """Full /v1/meminfo doc from a FrontendManager-like state, summed over all TP ranks.
    Never raises: a field the backend never sent (older build, still loading) reports 0."""
    cfg = getattr(state, "config", None)
    stats = getattr(state, "stats", None)
    live0 = _i(stats, "vram_bytes")
    ranks_in = _rank_payloads(state)

    ranks = []
    totals = {k: 0 for k in ("device_total_bytes", "weights_bytes", "overhead_bytes",
                             "pools_bytes", "cache_budget_bytes", "free_after_weights_bytes")}
    pool_tot = {d[0]: {"bytes": 0, "units": 0, "unit_cost_sum": 0,
                       "units_by_rank": [], "costs_by_rank": []} for d in _POOL_DEFS}
    for rank, m in ranks_in:
        pools = m.get("pools") or {}
        pbytes = {}
        for key, _label, _uk, _pk, _unit, _ck in _POOL_DEFS:
            units, cost, nbytes = _pool_alloc(pools, m, cfg, key)
            pbytes[key] = nbytes
            agg = pool_tot[key]
            agg["bytes"] += nbytes
            agg["units"] = max(agg["units"], units)  # logical count is per-replica, not summed
            agg["unit_cost_sum"] += cost
            # Under an uneven (bandwidth-weighted) shard the ranks disagree about what one
            # unit costs, so the scalars above cannot be multiplied back into `bytes`.
            agg["units_by_rank"].append(units)
            agg["costs_by_rank"].append(cost)
        row = {"rank": rank, "gpus": list(m.get("gpus") or []),
               "device_total_bytes": _i(m, "device_total_bytes"),
               "weights_bytes": _i(m, "weights_bytes"),
               "nonpool_overhead_bytes": _i(m, "nonpool_overhead_bytes"),
               "cache_budget_bytes": _i(m, "cache_budget_bytes"),
               "free_after_weights_bytes": _i(m, "free_vram_bytes"),
               "pools": pbytes, "pool_total_bytes": sum(pbytes.values()),
               # Never-sampled (0) reads as "not measured", not as a measurement of zero.
               "live_bytes": live0 if rank == 0 and live0 > 0 else None}
        residual = 0
        if rank == 0 and live0 > 0:
            residual = max(0, live0 - row["weights_bytes"] - row["pool_total_bytes"]
                           - row["nonpool_overhead_bytes"])
        row["residual_bytes"] = residual
        ranks.append(row)
        totals["device_total_bytes"] += row["device_total_bytes"]
        totals["weights_bytes"] += row["weights_bytes"]
        totals["overhead_bytes"] += row["nonpool_overhead_bytes"]
        totals["pools_bytes"] += row["pool_total_bytes"]
        totals["cache_budget_bytes"] += row["cache_budget_bytes"]
        totals["free_after_weights_bytes"] += row["free_after_weights_bytes"]

    # Rank0 live gauges as fill fractions priced at the summed all-rank allocation: exact
    # under symmetric TP sharding, an estimate where shards are deliberately uneven (MoE).
    def _ratio(num: str, den: str):
        d = _i(stats, den)
        return _i(stats, num) / d if d > 0 else None

    ratios = {"kv": (_ratio("kv_used_pages", "kv_total_pages"),
                     _ratio("kv_cached_pages", "kv_total_pages")),
              "swa": (_ratio("swa_used_tokens", "swa_total_tokens"),
                      _ratio("swa_cached_tokens", "swa_total_tokens")),
              # MoE has no kept set (every filled slot is evictable), so its split is the
              # last forward's working set against the warm rest of the fill.
              "moe": _moe_ratios(stats),
              "mamba": (_ratio("mamba_used_slots", "mamba_total_slots"),
                        _ratio("mamba_cached_slots", "mamba_total_slots"))}
    pools_out = []
    for key, label, _uk, _pk, unit, _ck in _POOL_DEFS:
        agg = pool_tot[key]
        if agg["bytes"] <= 0:
            continue
        r_used, r_cached = ratios.get(key, (None, None))
        pinned = int(round(r_used * agg["bytes"])) if r_used is not None else None
        cached = int(round(r_cached * agg["bytes"])) if r_cached is not None else None
        measured = pinned is not None or cached is not None
        pools_out.append({"key": key, "label": label, "unit": unit,
                          "units": agg["units"], "unit_bytes": agg["unit_cost_sum"],
                          "units_by_rank": agg["units_by_rank"],
                          "unit_costs_by_rank": agg["costs_by_rank"],
                          "bytes": agg["bytes"],
                          # Headline occupancy = kept + evictable warm cache (what the pool
                          # actually holds), clamped to the allocation.
                          "used_bytes": min(agg["bytes"], (pinned or 0) + (cached or 0))
                          if measured else None,
                          "pinned_bytes": pinned, "cached_bytes": cached,
                          # "split": kept + evictable warm cache (KV / GDN / SWA);
                          # "working_set": nothing is kept from eviction, so the dark part is
                          # the last forward's working set instead (MoE).
                          "det_mode": "working_set" if key == "moe" else "split",
                          "used_mode": "rank0" if measured else "none"})

    residual = sum(r_["residual_bytes"] for r_ in ranks)
    totals["residual_bytes"] = residual
    totals["allocated_bytes"] = (totals["weights_bytes"] + totals["pools_bytes"]
                                 + totals["overhead_bytes"])
    totals["used_est_bytes"] = totals["allocated_bytes"] + residual
    totals["free_est_bytes"] = max(0, totals["device_total_bytes"] - totals["used_est_bytes"])

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    tp_size = max(len(ranks_in), _i(cfg, "tp_size"),
                  max((_i(m, "tp_size") for _, m in ranks_in), default=0))
    return {
        "model": getattr(cfg, "served_model_name", None),
        "uptime_s": uptime_s,
        "tp_size": tp_size,
        "scope": "summed over TP ranks (live gauges measured on rank0)",
        "totals": totals,
        "pools": pools_out,
        "ranks": ranks,
        "live": {"rank0_vram_bytes": live0,
                 "kv_used_pages": _i(stats, "kv_used_pages"),
                 "kv_cached_pages": _i(stats, "kv_cached_pages"),
                 "kv_total_pages": _i(stats, "kv_total_pages"),
                 "mamba_used_slots": _i(stats, "mamba_used_slots"),
                 "mamba_cached_slots": _i(stats, "mamba_cached_slots"),
                 "mamba_total_slots": _i(stats, "mamba_total_slots"),
                 "moe_used_slots": _i(stats, "moe_used_slots"),
                 "moe_active_slots": _i(stats, "moe_active_slots"),
                 "moe_total_slots": _i(stats, "moe_total_slots")},
    }


MEMINFO_HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FreeToken GPU メモリ</title>
<style>
:root{color-scheme:dark}
body{background:#0b0f14;color:#d9e2ec;font:14px/1.45 "Segoe UI Variable Text",system-ui,"Noto Sans JP",sans-serif;margin:22px auto;max-width:1080px;padding:0 18px}
h1{font-size:16px;font-weight:600;margin:0 0 2px}#sub{color:#7c8a99;font-size:12px;margin-bottom:16px}
#err{color:#f87171;display:none;margin:8px 0}
.card{background:#111721;border:1px solid #1c2530;border-radius:8px;padding:14px 16px}
.big .head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.big .head b{font-size:30px;font-weight:600;letter-spacing:-.5px}
.big .head span{color:#9fb0c0;font-size:13px}
.stack{display:flex;height:16px;border-radius:4px;overflow:hidden;margin:12px 0 8px;background:#1c2530}
.stack i{height:100%}
.chips{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:12px;color:#9fb0c0}
.chips i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
.chips b{color:#d9e2ec;font-weight:600}
.kvline{font-size:12px;color:#9fb0c0;margin-top:10px;border-top:1px solid #1c2530;padding-top:8px}
.kvline b{color:#d9e2ec}
h2{font-size:13px;font-weight:600;color:#c3cfdb;margin:24px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.card h3{font-size:13px;font-weight:600;margin:0 0 10px;color:#c3cfdb}
.card .val{display:flex;align-items:baseline;gap:8px}
.card .val b{font-size:22px;font-weight:600}
.card .val span{color:#9fb0c0;font-size:12px}
.bar{height:8px;background:#1c2530;border-radius:4px;overflow:hidden;margin:10px 0 8px;display:flex}
.bar i{height:100%}
.det{font-size:12px;color:#9fb0c0}
.det b{color:#c9d5e1;font-weight:600}
.warn{color:#f87171}
table.gpus{border-collapse:collapse;font-size:12px;width:100%}
.gpus th,.gpus td{padding:5px 10px;border-bottom:1px solid #1c2530;text-align:right;font-variant-numeric:tabular-nums}
.gpus th{color:#9fb0c0;font-weight:400;border-bottom-color:#2a3644}
.gpus th:first-child,.gpus td:first-child,.gpus th:nth-child(2),.gpus td:nth-child(2){text-align:left}
.note{font-size:11px;color:#7c8a99;margin:14px 0 0}
</style></head><body>
<h1>GPU メモリ</h1>
<div id="sub">読み込み中…</div>
<div id="err"></div>
<div id="app"></div>
<script>
const GIB=1073741824;
const COL={weights:'#64748b',kv:'#22c55e',swa:'#38bdf8',moe:'#f59e0b',mamba:'#a78bfa',
 overhead:'#94a3b8',residual:'#3f4a56',free:'#1a222c'};
const fmt=b=>b==null||isNaN(b)||b<0?'-':(Math.abs(b)>=GIB?(b/GIB).toFixed(2)+' GiB':(b/1048576).toFixed(0)+' MiB');
// Per-unit prices need sub-MiB digits: a KV token costs ~13 KiB and fmt() would print "0 MiB".
const fmtu=b=>b==null||isNaN(b)||b<0?'-':(b>=GIB?(b/GIB).toFixed(2)+' GiB':b>=1048576?(b/1048576).toFixed(1)+' MiB':b>=1024?(b/1024).toFixed(1)+' KiB':b+' B');
function gpuname(r){return (r.gpus||[]).map(g=>'GPU'+g.index+' '+g.name).join(' / ')||'-';}
function poolcard(p){
 const cap=p.bytes,u=p.used_bytes,pin=p.pinned_bytes,ca=p.cached_bytes;
 const has=u!=null,f=has?Math.min(100,100*u/cap):0;
 const pct=has?(' / '+Math.round(100*u/cap)+'%'):'';
 const fpin=has&&cap>0?Math.min(100,100*(pin||0)/cap):0;
 const fca=has?Math.max(0,f-fpin):0;
 const bar=`<div class="bar"><i style="width:${fpin.toFixed(1)}%;background:${COL[p.key]}"></i><i style="width:${fca.toFixed(1)}%;background:${COL[p.key]};opacity:.45"></i></div>`;
 const ws=p.det_mode==='working_set',p0=pin||0,c0=ca||0;
 const det=has?(p0>0
   ?`${ws?'うち現役ステップが使用':'うちアクティブ(排他)'} <b>${fmt(p0)}</b> ・ キャッシュ <b>${fmt(c0)}</b>（退避可） ／ rank0比率×全GPU確保`
   :(c0>0?`待機中：全 <b>${fmt(c0)}</b> はキャッシュとしてプールに残存 ・ 次リクエストで再利用`
     :'プール空 — まだ履歴は蓄積されていません'))
   :`確保 <b>${fmt(cap)}</b>（常駐） ・ ゲージ未稼働`;
 const per=p.units_by_rank||[],costs=p.unit_costs_by_rank||[];
 const skew=per.length>1&&(new Set(per).size>1||new Set(costs).size>1);
 const units=skew
  ?`<div class="det">${per.map((n,i)=>`${n.toLocaleString()}×${fmtu(costs[i])}`).join(' + ')} ${p.unit}（rank別の枠×単価・積の和が上限）</div>`
  :`<div class="det">${(p.units||0).toLocaleString()} ${p.unit} × ${fmtu(p.unit_bytes)}/u</div>`;
 return `<div class="card"><h3>${p.label}</h3>
 <div class="val"><b>${has?fmt(u):'—'}</b><span>${has?'蓄積 / 上限 '+fmt(cap)+pct:'上限 '+fmt(cap)+' ／ 蓄積 未計測'}</span></div>
 ${bar}<div class="det">${det}</div>${units}</div>`;}
function render(d){
 const T=d.totals||{},P=d.pools||[],R=d.ranks||[],tot=T.device_total_bytes||0;
 const PB={};P.forEach(p=>PB[p.key]=p.bytes);
 document.getElementById('sub').textContent=
  (d.model||'')+' ｜ '+R.length+' GPU ・ TP合算 ｜ uptime '+(d.uptime_s||0)+'s ｜ 自動更新 3s';
 const segs=[{k:'weights',label:'重み',bytes:T.weights_bytes},
  {k:'kv',label:'KV',bytes:PB.kv},{k:'swa',label:'SWA',bytes:PB.swa},
  {k:'moe',label:'MoE',bytes:PB.moe},{k:'mamba',label:'GDN',bytes:PB.mamba},
  {k:'overhead',label:'予備領域',bytes:T.overhead_bytes},
  {k:'residual',label:'残余(rank0実測)',bytes:T.residual_bytes},
  {k:'free',label:'空き(推定)',bytes:T.free_est_bytes}]
  .map(s=>({label:s.label,color:COL[s.k],bytes:s.bytes||0})).filter(s=>s.bytes>0);
 const stack=tot>0?`<div class="stack">${segs.map(s=>`<i style="width:${(100*s.bytes/tot).toFixed(2)}%;background:${s.color}" title="${s.label} ${fmt(s.bytes)}"></i>`).join('')}</div>`:'';
 const chips=segs.map(s=>`<span><i style="background:${s.color}"></i>${s.label} <b>${fmt(s.bytes)}</b></span>`).join('');
 const B=T.cache_budget_bytes,PT=T.pools_bytes||0,over=B>0&&PT>B;
 const top=`<div class="card big">
 <div class="head"><b>${fmt(T.used_est_bytes)}</b><span>使用中（推定） ／ 全GPU合計 <b style="color:#d9e2ec">${fmt(tot)}</b> ｜ rank0実測 ${(d.live||{}).rank0_vram_bytes?fmt(d.live.rank0_vram_bytes):'-'}</span></div>
 ${stack}<div class="chips">${chips}</div>
 <div class="kvline">プール確保 <b>${fmt(PT)}</b> ／ cache budget <b>${fmt(B)}</b> <span class="${over?'warn':''}">(${B>0?Math.round(100*PT/B):'-'}%${over?' 超過 — rebuild fit 上限を厳守せず':''})</span> ・ 重み後空き ${fmt(T.free_after_weights_bytes)}</div></div>`;
 const cards=P.length?`<div class="grid">${P.map(poolcard).join('')}</div>`:'<div class="det">プール未配載</div>';
 const grows=R.map(r=>`<tr><td>${r.rank}</td><td>${gpuname(r)}</td><td>${fmt(r.device_total_bytes)}</td><td>${fmt(r.weights_bytes)}</td>
  <td>${fmt((r.pools||{}).kv)}</td><td>${fmt((r.pools||{}).swa)}</td><td>${fmt((r.pools||{}).moe)}</td><td>${fmt((r.pools||{}).mamba)}</td>
  <td>${fmt(r.nonpool_overhead_bytes)}</td><td>${fmt(r.cache_budget_bytes)}</td><td>${r.live_bytes!=null?fmt(r.live_bytes):'-'}</td></tr>`).join('');
 const gtable=`<table class="gpus"><tr><th>Rank</th><th>GPU</th><th>総計</th><th>重み</th><th>KV</th><th>SWA</th><th>MoE</th><th>GDN</th><th>予備</th><th>予算</th><th>live</th></tr>${grows}</table>`;
 const note='使用中=確保済み(重み+プール+予備)+残余(rank0のみ実測)。各プールの「蓄積」=アクティブ(現役リクエストが排他使用中)+キャッシュ(生成後の履歴はここに自動で移る。破棄ではなく確保されたまま、次リクエストのプレフィックスヒットに使える)。barは実色=アクティブ・明るい分=キャッシュ。実VRAMは確保時点で確保分が常時在位するため推移しない。live/蓄積はrank0のゲージ比×全GPU確保分(対称シャードで厳密)。MoEプールに排他確保はなく、実色=直前のforwardが読んだスロット(残りもLRUがいつでも追い出せる)。枠×単価の行が2項以上ならTPシャードが非対称で、各項の積の和が上限。JSON: <code>/v1/meminfo</code>';
 document.getElementById('app').innerHTML=top
  +'<h2>キャッシュプール — 蓄積と上限</h2>'+cards
  +'<h2>GPU ごとの割り当て</h2>'+gtable
  +'<div class="note">'+note+'</div>';
 document.getElementById('err').style.display='none';}
async function tick(){try{const r=await fetch('/v1/meminfo');if(!r.ok)throw new Error('HTTP '+r.status);render(await r.json());}
 catch(e){const el=document.getElementById('err');el.textContent='取得失敗: '+e.message;el.style.display='block';}}
tick();setInterval(tick,3000);
</script></body></html>"""
