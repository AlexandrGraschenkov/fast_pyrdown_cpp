//
//  eigen_pyrdown.cpp
//  fast_pyrdown
//
//  Created by Alexander Graschenkov on 22.12.2022.
//

#include "eigen_pyrdown.hpp"
#include <Eigen/Dense>

//using PrecisionAllocator = Eigen::aligned_allocator<Precision>;

template <typename Precision>
void eigenPyrdown(const Precision *image, int height, int width, std::vector<Precision> &out) {
    using EigenMat = Eigen::Matrix<Precision, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const EigenMat> image_map(image, height, width);
    out.resize((height/2) * (width/2));
    Eigen::Map<EigenMat> resized_image_map(out.data(), height / 2, width / 2);
    
    resized_image_map = 0.25f * (image_map(Eigen::seq(Eigen::fix<0>, height - 1, Eigen::fix<2>),
                                           Eigen::seq(Eigen::fix<0>, width - 1, Eigen::fix<2>)) +
                                 image_map(Eigen::seq(Eigen::fix<1>, height, Eigen::fix<2>),
                                           Eigen::seq(Eigen::fix<1>, width, Eigen::fix<2>)) +
                                 image_map(Eigen::seq(Eigen::fix<0>, height - 1, Eigen::fix<2>),
                                           Eigen::seq(Eigen::fix<1>, width, Eigen::fix<2>)) +
                                 image_map(Eigen::seq(Eigen::fix<1>, height, Eigen::fix<2>),
                                           Eigen::seq(Eigen::fix<0>, width - 1, Eigen::fix<2>)));
}


//template std::vector<uint8_t> eigenPyrdown(const std::vector<uint8_t> &image, int height, int width);
//template void eigenPyrdown(const uint8_t *image, int height, int width, std::vector<uint8_t> &out);
template void eigenPyrdown(const float *image, int height, int width, std::vector<float> &out);
