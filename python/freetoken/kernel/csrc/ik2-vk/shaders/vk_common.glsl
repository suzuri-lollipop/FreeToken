// Shared BDA buffer types, dtype IO helpers, SITU activation, and warp-coalesced
// quantized dot products (subgroup size 32) for ik2-vk.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32Buf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer U16Buf { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer F32Buf { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer I32Buf { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer F32Out { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) writeonly buffer U16Out { uint16_t v[]; };

// io_dtype codes: 0 = F32, 1 = BF16, 2 = F16
float read_val(uint64_t addr, int dt, uint idx) {
    if (dt == 0) return F32Buf(addr).v[idx];
    uint raw = uint(U16Buf(addr).v[idx]);
    if (dt == 1) return uintBitsToFloat(raw << 16u);
    return unpackHalf2x16(raw).x;
}

uint16_t f32_to_bf16(float f) {
    uint u = floatBitsToUint(f);
    uint lsb = (u >> 16u) & 1u;
    u += 0x7fffu + lsb;
    return uint16_t(u >> 16u);
}

void write_val(uint64_t addr, int dt, uint idx, float val) {
    if (dt == 0) {
        F32Out(addr).v[idx] = val;
    } else if (dt == 1) {
        U16Out(addr).v[idx] = f32_to_bf16(val);
    } else {
        U16Out(addr).v[idx] = uint16_t(packHalf2x16(vec2(val, 0.0)) & 0xFFFFu);
    }
}

uint getb16(U16Buf buf, uint byte_off) {
    uint word = uint(buf.v[byte_off >> 1u]);
    return (byte_off & 1u) != 0u ? (word >> 8u) : (word & 0xFFu);
}

// ggml NVFP4 scale byte: half of E4M3 (2^(e-8)), 0 and 0x7F -> 0
float e4m3_half(uint v) {
    if (v == 0x7Fu || v == 0u) return 0.0;
    uint e = (v >> 3u) & 0xFu, m = v & 7u;
    if (e == 0u) return float(m) * 9.765625e-4;
    return uintBitsToFloat(((e + 119u) << 23u) | (m << 20u));
}

// doubled E2M1 code value, arithmetically (no register-spilling LUT)
float kv_e2m1(uint c) {
    uint m = c & 7u;
    float v = m < 4u ? float(m) : (m < 6u ? float(2u * m - 4u) : float(4u * m - 16u));
    return (c & 8u) != 0u ? -v : v;
}

// SITU: (gcap * tanh(g / gcap)) * sigmoid(g) * (ucap * tanh(u / ucap))
float situ_act(float g, float u, float gcap, float ucap) {
    float tg = gcap * tanh(g / gcap);
    float sg = 1.0 / (1.0 + exp(-g));
    float tu = ucap * tanh(u / ucap);
    return tg * sg * tu;
}

// Warp-coalesced dot product of weight row `row` (type `ty`, length `K`) with
// activation vector at `x_addr + x_base` (dtype `x_dt`). Caller must invoke with
// full 32-lane subgroup; lane 0 receives the exact reduced sum via subgroupAdd.
float warp_dot_row(uint64_t w_addr, int ty, uint row, int K, uint64_t x_addr, int x_dt, uint x_base) {
    uint lane = gl_LocalInvocationID.x;
    float acc = 0.0;

    if (ty == 40) {  // NVFP4: 36B = 9 u32 per 64 vals; 8 lanes/block -> 4 blocks (256 vals)/step
        U32Buf W = U32Buf(w_addr);
        uint n_blk = uint(K) >> 6u;
        uint row_u32 = row * n_blk * 9u;
        uint b_off = lane >> 3u;
        uint qw = lane & 7u;
        uint sb = qw >> 1u;
        uint hw = qw & 1u;
        for (uint blk = b_off; blk < n_blk; blk += 4u) {
            uint bu = row_u32 + blk * 9u;
            uint sw = W.v[bu];
            uint qbits = W.v[bu + 1u + qw];
            float sc = e4m3_half((sw >> (sb * 8u)) & 0xFFu);
            uint bk = x_base + blk * 64u + sb * 16u + hw * 4u;
            uint b0 = qbits & 0xFFu;
            uint b1 = (qbits >> 8u) & 0xFFu;
            uint b2 = (qbits >> 16u) & 0xFFu;
            uint b3 = qbits >> 24u;
            float s = 0.0;
            s = fma(kv_e2m1(b0 & 0xFu), read_val(x_addr, x_dt, bk + 0u), s);
            s = fma(kv_e2m1(b1 & 0xFu), read_val(x_addr, x_dt, bk + 1u), s);
            s = fma(kv_e2m1(b2 & 0xFu), read_val(x_addr, x_dt, bk + 2u), s);
            s = fma(kv_e2m1(b3 & 0xFu), read_val(x_addr, x_dt, bk + 3u), s);
            s = fma(kv_e2m1(b0 >> 4u),  read_val(x_addr, x_dt, bk + 8u), s);
            s = fma(kv_e2m1(b1 >> 4u),  read_val(x_addr, x_dt, bk + 9u), s);
            s = fma(kv_e2m1(b2 >> 4u),  read_val(x_addr, x_dt, bk + 10u), s);
            s = fma(kv_e2m1(b3 >> 4u),  read_val(x_addr, x_dt, bk + 11u), s);
            acc = fma(s, sc, acc);
        }
    } else if (ty == 8) {  // Q8_0: 34B = 17 u16 per 32 vals; 8 lanes/block -> 4 blocks (128 vals)/step
        U16Buf W = U16Buf(w_addr);
        uint n_blk = uint(K) >> 5u;
        uint row_u16 = row * n_blk * 17u;
        uint b_off = lane >> 3u;
        uint qw = lane & 7u;
        for (uint blk = b_off; blk < n_blk; blk += 4u) {
            uint bu = row_u16 + blk * 17u;
            float d = unpackHalf2x16(uint(W.v[bu])).x;
            uint p0 = uint(W.v[bu + 1u + qw * 2u]);
            uint p1 = uint(W.v[bu + 2u + qw * 2u]);
            int pack = int(p0 | (p1 << 16u));
            uint bk = x_base + blk * 32u + qw * 4u;
            float s = 0.0;
            s = fma(float(bitfieldExtract(pack, 0, 8)),  read_val(x_addr, x_dt, bk + 0u), s);
            s = fma(float(bitfieldExtract(pack, 8, 8)),  read_val(x_addr, x_dt, bk + 1u), s);
            s = fma(float(bitfieldExtract(pack, 16, 8)), read_val(x_addr, x_dt, bk + 2u), s);
            s = fma(float(bitfieldExtract(pack, 24, 8)), read_val(x_addr, x_dt, bk + 3u), s);
            acc = fma(s, d, acc);
        }
    } else if (ty == 30) {  // BF16: 2 vals per u32 -> 64 vals/step
        U32Buf W = U32Buf(w_addr);
        uint n_u32 = uint(K) >> 1u;
        uint row_u32 = row * n_u32;
        for (uint i = lane; i < n_u32; i += 32u) {
            uint u = W.v[row_u32 + i];
            float w0 = uintBitsToFloat(u << 16u);
            float w1 = uintBitsToFloat(u & 0xFFFF0000u);
            uint bk = x_base + i * 2u;
            acc = fma(w0, read_val(x_addr, x_dt, bk), acc);
            acc = fma(w1, read_val(x_addr, x_dt, bk + 1u), acc);
        }
    } else if (ty == 0) {  // F32: 32 vals/step
        F32Buf W = F32Buf(w_addr);
        uint row_f = row * uint(K);
        for (uint i = lane; i < uint(K); i += 32u) {
            acc = fma(W.v[row_f + i], read_val(x_addr, x_dt, x_base + i), acc);
        }
    } else if (ty == 2) {  // Q4_0: 18B = 9 u16 per 32 vals; 8 lanes/block -> 4 blocks/step
        U16Buf W = U16Buf(w_addr);
        uint n_blk = uint(K) >> 5u;
        uint row_u16 = row * n_blk * 9u;
        uint b_off = lane >> 3u;
        uint qw = lane & 7u;
        for (uint blk = b_off; blk < n_blk; blk += 4u) {
            uint bu = row_u16 + blk * 9u;
            float d = unpackHalf2x16(uint(W.v[bu])).x;
            uint p16 = uint(W.v[bu + 1u + qw]);
            uint b0 = p16 & 0xFFu, b1 = p16 >> 8u;
            uint bk = x_base + blk * 32u + qw * 2u;
            float s = 0.0;
            s = fma(float(int(b0 & 0xFu) - 8), read_val(x_addr, x_dt, bk + 0u), s);
            s = fma(float(int(b1 & 0xFu) - 8), read_val(x_addr, x_dt, bk + 1u), s);
            s = fma(float(int(b0 >> 4u) - 8),  read_val(x_addr, x_dt, bk + 16u), s);
            s = fma(float(int(b1 >> 4u) - 8),  read_val(x_addr, x_dt, bk + 17u), s);
            acc = fma(s, d, acc);
        }
    } else {  // Q6_K (ty == 14): 210B = 105 u16 per 256 vals; 32 lanes/block
        U16Buf W = U16Buf(w_addr);
        uint n_blk = uint(K) >> 8u;
        uint row_b = row * n_blk * 210u;
        uint l = lane;
        for (uint blk = 0u; blk < n_blk; ++blk) {
            uint bb = row_b + blk * 210u;
            float d = unpackHalf2x16(uint(W.v[(bb + 208u) >> 1u])).x;
            float s = 0.0;
            for (uint n = 0u; n < 2u; ++n) {
                uint qh_b = getb16(W, bb + 128u + n * 32u + l);
                uint ql0 = getb16(W, bb + n * 64u + l);
                uint ql1 = getb16(W, bb + n * 64u + l + 32u);
                uint sc_base = bb + 192u + n * 8u + (l >> 4u);
                uint kb = x_base + blk * 256u + n * 128u + l;
                for (uint qd = 0u; qd < 4u; ++qd) {
                    uint lo = (qd & 1u) != 0u ? ql1 : ql0;
                    uint nib = qd >= 2u ? (lo >> 4u) : (lo & 0xFu);
                    uint qh = (qh_b >> (2u * qd)) & 3u;
                    int q = int(nib | (qh << 4u)) - 32;
                    int sc = int(getb16(W, sc_base + 2u * qd));
                    if (sc >= 128) sc -= 256;
                    s = fma(float(sc * q), read_val(x_addr, x_dt, kb + qd * 32u), s);
                }
            }
            acc = fma(s, d, acc);
        }
    }
    return subgroupAdd(acc);
}
