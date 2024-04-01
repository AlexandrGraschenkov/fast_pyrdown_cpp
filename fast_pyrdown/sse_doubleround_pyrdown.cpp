//
//  sse_doubleround_pyrdown.cpp
//  fast_pyrdown
//
//  Created by Alexander Graschenkov on 22.12.2022.
//

#include "sse_doubleround_pyrdown.hpp"
#include "sse2neon.h"
//#include <emmintrin.h>
//#include <immintrin.h>
#include <cmath>
#include <iostream>

using namespace std;


static __inline __m128i average2RowsSingle(const uint8_t* __restrict__ src, size_t srcStep) {
    __m128i v0 = _mm_load_si128((const __m128i *)src);
    __m128i v1 = _mm_load_si128((const __m128i *)&src[srcStep]);
    return _mm_avg_epu8(v0, v1);
}

// SSSE3 version
// I used `__restrict__` to give the compiler more flexibility in unrolling
void average2Rows(const uint8_t* __restrict__ src,
                  uint8_t*__restrict__ dst,
                  size_t srcStep,
                  size_t size)
{
    const __m128i vk1 = _mm_set1_epi8(1);
    const __m128i add2 = _mm_set1_epi16(2);
    size_t dstsize = size/2;
    for (size_t i = 0; i < dstsize - 15; i += 16)
    {
        const size_t ii = i*2;
        if (true) {
            // https://stackoverflow.com/a/45564565/820795
            __m128i left  = average2RowsSingle(src+ii, srcStep);
            __m128i right = average2RowsSingle(src+ii+16, srcStep);
            
            __m128i w0 = _mm_maddubs_epi16(left, vk1);        // unpack and horizontal add
            __m128i w1 = _mm_maddubs_epi16(right, vk1);
            w0 = _mm_srli_epi16(w0, 1);                     // divide by 2
            w1 = _mm_srli_epi16(w1, 1);
            w0 = _mm_packus_epi16(w0, w1);                  // pack
            
            _mm_storeu_si128((__m128i *)&dst[i], w0);
        } else {
            // more accurate
            __m128i v0 = _mm_load_si128((const __m128i *)&src[ii]);
            __m128i v1 = _mm_load_si128((const __m128i *)&src[ii + 16]);
            __m128i v2 = _mm_load_si128((const __m128i *)&src[ii+srcStep]);
            __m128i v3 = _mm_load_si128((const __m128i *)&src[ii+srcStep + 16]);
            
            v0 = _mm_maddubs_epi16(v0, vk1);
            v2 = _mm_maddubs_epi16(v2, vk1);
            v0 = _mm_add_epi16(_mm_add_epi16(v0, v2), add2);
            v0 = _mm_srli_epi16(v0, 2);                     // divide by 4
            
            v1 = _mm_maddubs_epi16(v1, vk1);
            v3 = _mm_maddubs_epi16(v3, vk1);
            v1 = _mm_add_epi16(_mm_add_epi16(v1, v3), add2);
            v1 = _mm_srli_epi16(v1, 2);                     // divide by 4
            
            v0 = _mm_packus_epi16(v0, v1);                  // pack
            _mm_storeu_si128((__m128i *)&dst[i], v0);
        }
    }
}

static __inline __m128i average4RowsSingle(const uint8_t* __restrict__ src, size_t srcStep) {
    __m128i v0 = _mm_load_si128((const __m128i *)src);
    __m128i v1 = _mm_load_si128((const __m128i *)&src[srcStep]);
    __m128i v2 = _mm_load_si128((const __m128i *)&src[srcStep*2]);
    __m128i v3 = _mm_load_si128((const __m128i *)&src[srcStep*3]);
    v0 = _mm_avg_epu8(v0, v3);
    v1 = _mm_avg_epu8(v1, v2);
    return _mm_avg_epu8(v0, v1);
}

void average4Rows(const uint8_t* __restrict__ src,
                  uint8_t*__restrict__ dst,
                  size_t srcStep,
                  size_t size)
{
    const __m128i vk1 = _mm_set1_epi8(1);
    size_t dstsize = size/4;
    for (size_t i = 0; i < dstsize - 15; i += 16)
    {
        size_t ii = i*4;
        __m128i left = average4RowsSingle(src+ii, srcStep);
        __m128i right = average4RowsSingle(src+ii+16, srcStep);
        
        __m128i w0 = _mm_maddubs_epi16(left, vk1);        // unpack and horizontal add
        __m128i w1 = _mm_maddubs_epi16(right, vk1);
        __m128i res = _mm_hadds_epi16(w0, w1);
        
        left = average4RowsSingle(src+ii+32, srcStep);
        right = average4RowsSingle(src+ii+48, srcStep);
        
        w0 = _mm_maddubs_epi16(left, vk1);        // unpack and horizontal add
        w1 = _mm_maddubs_epi16(right, vk1);
        
        res = _mm_packus_epi16(_mm_srli_epi16(res, 2),
                               _mm_srli_epi16(_mm_hadds_epi16(w0, w1), 2));                  // pack
        _mm_storeu_si128((__m128i *)&dst[i], res);
    }
}


static __inline __m128i divide3_16i(__m128i v) {
    v = _mm_mullo_epi16(v, _mm_set1_epi16(43)); // sum *= (128 / 3); // 43
    return v = _mm_srli_epi16(v,7); // sum /= 128
}

static __inline __m128i vertical_avg3_i16_fast(const __m128i &a, const __m128i &b, const __m128i &c) {
    auto sum = _mm_add_epi16(a, _mm_add_epi16(b, c));
    return divide3_16i(sum);
}

static __inline __m128i vertical_avg3_i16_slow(const __m128i &a, const __m128i &b, const __m128i &c) {
    // stackoverflow magic to multiply coef
    const __m128i coef = _mm_set1_epi16((short)(32768.0/3 + 0.5));
    __m128i sum = _mm_add_epi16(_mm_add_epi16(
                                _mm_mulhrs_epi16(_mm_slli_epi16(a, 6), coef),
                                _mm_mulhrs_epi16(_mm_slli_epi16(b, 6), coef)),
                                _mm_mulhrs_epi16(_mm_slli_epi16(c, 6), coef));
    return _mm_srli_epi16(sum, 6);
}

template <bool Fast>
static __inline __m128i horisontal_avg3_unpack(const __m128i &v_in) {
    const signed char o = -1; // make shuffle zero
    const __m128i vec_r_i16 = _mm_shuffle_epi8(v_in, _mm_set_epi8(o,o,o,o,o,o, o,12, o,9,  o,6, o,3, o,0));
    const __m128i vec_g_i16 = _mm_shuffle_epi8(v_in, _mm_set_epi8(o,o,o,o,o,o, o,13, o,10, o,7, o,4, o,1));
    const __m128i vec_b_i16 = _mm_shuffle_epi8(v_in, _mm_set_epi8(o,o,o,o,o,o, o,14, o,11, o,8, o,5, o,2));
    
    // my impl
    if (Fast) {
        return vertical_avg3_i16_fast(vec_r_i16, vec_g_i16, vec_b_i16);
    } else {
        return vertical_avg3_i16_slow(vec_r_i16, vec_g_i16, vec_b_i16);
    }
}

template <bool Fast>
static __m128i average3RowsSingle(const uint8_t *__restrict src, size_t srcStep) {
    __m128i r0 = _mm_load_si128((const __m128i *)src);
    __m128i r1 = _mm_load_si128((const __m128i *)&src[srcStep]);
    __m128i r2 = _mm_load_si128((const __m128i *)&src[srcStep*2]);
    
    r0 = horisontal_avg3_unpack<Fast>(r0);
    r1 = horisontal_avg3_unpack<Fast>(r1);
    r2 = horisontal_avg3_unpack<Fast>(r2);
    if (Fast) {
        return vertical_avg3_i16_fast(r0, r1, r2);
    } else {
        return vertical_avg3_i16_slow(r0, r1, r2);
    }
}

template <bool Fast>
void average3Rows(const uint8_t* __restrict__ src,
                  uint8_t*__restrict__ dst,
                  size_t srcStep,
                  size_t size,
                  bool debug = false)
{
    const signed char o = -1; // make shuffle zero
    const size_t dstsize = size/3;
    for (size_t i = 0; i < dstsize - 9; i += 10)
    {
        const size_t ii = 3*i;
        __m128i r0 = average3RowsSingle<Fast>(src+ii, srcStep);
        __m128i r1 = average3RowsSingle<Fast>(src+ii+15, srcStep);
//        __m128i r2 = average3RowsSingle(src+ii+30, srcStep);
        
        r0 = _mm_packus_epi16(r0, r1);
        r0 = _mm_shuffle_epi8(r0, _mm_set_epi8(o, o,o,o,o,o, 12,11,10,9,8, 4,3,2,1,0));
//        r0 = _mm_packus_epi16(r0, _mm_setzero_si128());                  // pack
        
        _mm_storeu_si128((__m128i *)&dst[i], r0);
    }
}

// MARK: - interface

using namespace std;
void ssePyrdown2(const uint8_t *image, int height, int width, std::vector<uint8_t> &out) {
    const int half_width = width / 2;
    const int half_height = height / 2;
    out.resize(half_width * half_height);
    vector<uint8_t> &half_image = out;
    
    const int width_rest32 = width % 32;
    bool process_rest_manual_approach = width_rest32 > 0 && width_rest32 < 5;
    for (int r = 0; r < half_height; r++) {
        const int offset = r * 2 * width;
        const int half_offset = r * half_width;
        
        int process_width = ceilf(width / 32.f) * 32;
        if (r == half_height-1 || process_rest_manual_approach) {
            process_width = floorf(width / 32.f) * 32;
        }
        average2Rows(&image[offset], &half_image[half_offset], width, process_width);
        
        // process rest of line
        for (int c = process_width; c < width; c += 2) {
            out[half_offset + c/2] = ((int)image[offset+c] +
                                      (int)image[offset+c+1] +
                                      (int)image[offset+width+c] +
                                      (int)image[offset+width+c+1]) / 4;
        }
    }
}
void ssePyrdown3(const uint8_t *image, int height, int width, std::vector<uint8_t> &out) {
    const int result_width = width / 3;
    const int result_height = height / 3;
    out.resize(result_width * result_height);
    vector<uint8_t> &half_image = out;
    
    const int width_rest32 = width % 5;
    bool process_rest_manual_approach = width_rest32 > 0;
    for (int r = 0; r < result_height; r++) {
        const int offset = r * 3 * width;
        const int half_offset = r * result_width;
        
        int process_width = ceilf(width / 10.f) * 10;
        if (r == result_height-1 || process_rest_manual_approach) {
            process_width = floorf(width / 10.f) * 10;
        }
        average3Rows<true>(&image[offset], &half_image[half_offset], width, process_width);
        
        // process rest of line
        const uint8_t *in_data = image + offset;
        for (int c = process_width; c < width; c += 3) {
            out[half_offset + c/3] = ((int)in_data[c] + (int)in_data[c+1] + (int)in_data[c+2] +
                                      (int)in_data[width+c] + (int)in_data[width+c+1] + (int)in_data[width+c+2] +
                                      (int)in_data[width*2+c] + (int)in_data[width*2+c+1] + (int)in_data[width*2+c+2]) / 9;
        }
    }
}

void ssePyrdown4(const uint8_t *image, int height, int width, std::vector<uint8_t> &out) {
    const int quad_width = width / 4;
    const int quad_height = height / 4;
    out.resize(quad_width * quad_height);
    vector<uint8_t> &half_image = out;
    
    const int width_rest32 = width % 32;
    bool process_rest_manual_approach = width_rest32 > 0 && width_rest32 < 5;
    for (int r = 0; r < quad_height; r++) {
        const int offset = r * 4 * width;
        const int quad_offset = r * quad_width;
        
        int process_width = ceilf(width / 64.f) * 64;
        if (r == quad_height-1 || process_rest_manual_approach) {
            process_width = floorf(width / 64.f) * 64;
        }
        average4Rows(&image[offset], &half_image[quad_offset], width, process_width);
        
        // process rest of line
        for (int c = process_width; c < width; c += 4) {
            int32_t val = 0;
            for (int rr = 0; rr < 4; r++) {
                for (int cc = 0; cc < 4; cc++) {
                    val += image[offset + c + rr*width + cc];
                }
            }
            out[quad_offset + c/4] = val / 16;
        }
    }
}

// MARK: -----  Float
void average2RowsFloat(const float* __restrict__ src,
                  float*__restrict__ dst,
                  size_t srcWidth,
                  size_t size)
{
    const __m128i _f4 = _mm_set1_ps(4);
    const size_t dstsize = size/2;
    for (size_t i = 0; i < dstsize - 3; i += 4) {
        const size_t ii = i*2;
        /// load data and horizontal add neighbors
        /// in: [a1, a2, a3, a4], [b1, b2, b3, b4]
        /// out: [a1+a2, a3+a4, b1+b2, b3+b4]
        __m128 v0 = _mm_hadd_ps(_mm_load_ps(&src[ii]),
                                _mm_load_ps(&src[ii+4]));
        __m128 v1 = _mm_hadd_ps(_mm_load_ps(&src[ii+srcWidth]),
                                _mm_load_ps(&src[ii+srcWidth+4]));
        
        /// vertical add floats
        /// in: [a1, a2, a3, a4], [b1, b2, b3, b4]
        /// out: [a1+b1, a2+b2, a3+b3, a4+b4]
        v0 = _mm_add_ps(v0, v1);
        
        /// divide each element by 4
        v0 = _mm_div_ps(v0, _f4);
        
        _mm_store_ps(dst+i, v0);
    }
}

void ssePyrdownF(const float *image, int height, int width, std::vector<float> &out) {
    const int half_width = width / 2;
    const int half_height = height / 2;
    out.resize(half_width * half_height);
    for (int r = 0; r < half_height; r++) {
        const int offset = r * 2 * width;
        const int half_offset = r * half_width;
        
        const int process_width = (width / 4) * 4;
        average2RowsFloat(image+offset, &out[half_offset], width, process_width);
        for (int c = process_width; c < width; c += 2) {
//            out[half_offset + c/2] = 100;
            out[half_offset + c/2] = (image[offset+c] +
                                      image[offset+c+1] +
                                      image[offset+width+c] +
                                      image[offset+width+c+1]) * 0.25f;
        }
    }
}
