//
//  sse_doubleround_pyrdown.hpp
//  fast_pyrdown
//
//  Created by Alexander Graschenkov on 22.12.2022.
//

#ifndef sse_doubleround_pyrdown_hpp
#define sse_doubleround_pyrdown_hpp

#include <stdio.h>
#include <vector>

void ssePyrdown2(const uint8_t *image, int height, int width, std::vector<uint8_t> &out);
void ssePyrdown3(const uint8_t *image, int height, int width, std::vector<uint8_t> &out);
void ssePyrdown4(const uint8_t *image, int height, int width, std::vector<uint8_t> &out);

void ssePyrdownF(const float *image, int height, int width, std::vector<float> &out);

#endif /* sse_doubleround_pyrdown_hpp */
