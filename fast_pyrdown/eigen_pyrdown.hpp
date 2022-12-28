//
//  eigen_pyrdown.hpp
//  fast_pyrdown
//
//  Created by Alexander Graschenkov on 22.12.2022.
//

#ifndef eigen_pyrdown_hpp
#define eigen_pyrdown_hpp

#include <stdio.h>
#include <vector>

template <typename Precision>
void eigenPyrdown(const Precision *image, int height, int width, std::vector<Precision> &out);

#endif /* eigen_pyrdown_hpp */
