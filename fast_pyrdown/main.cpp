//
//  main.cpp
//  fast_pyrdown
//
//  Created by Alexander Graschenkov on 22.12.2022.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "eigen_pyrdown.hpp"
#include "sse_doubleround_pyrdown.hpp"
#include "rbenchmark.hpp"
#include "sse2neon.h"
#include <Eigen/Core>
#include <wchar.h>

using namespace cv;
using namespace std;



void manualResize(const cv::Mat &src, cv::Mat &dst, int timesN) {
    dst.create(src.rows / timesN, src.cols / timesN, CV_8UC1);
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            auto res = cv::mean(src(Rect(c*timesN, r*timesN, timesN, timesN)));
            dst.at<uint8_t>(r, c) = round(res[0]);
        }
    }
}

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
//    cout << Eigen::
    
////    vector<uint8_t> testData = {128, 0, 0, 0, 255, 0, 7, 8, 0, 10, 11, 0, 13, 14, 0, 16};
//    vector<uint8_t> testOut = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//    vector<uint8_t> testData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//
//    __m128 v = _mm_load_si128((const __m128i *)testData.data());
//    __m128 v2 = _mm_shuffle_epi8(v, _mm_set_epi8(-1,7,-1,6, -1,5,-1,4, -1,3,-1,2, -1,1,-1,1));
//    v2 = _mm_packs_epi16(v2, _mm_setzero_si128());                  // pack
//    v = _mm_cvtepu8_epi16(v);
//
//    _mm_stream_si128((__m128i *)testOut.data(), v2);
//    for (int i = 0; i < testOut.size(); i++) {
//        cout << i << " <> " << (int)testOut[i] << endl;
//    }
////    __m128 v = _mm_set_epi8(127, 0, 14, 13, 0, 11, 10, 0, 8, 7, 0, 0, 0, 0, 0, 0);
////    int v2 = _mm_movemask_epi8(v);
////    for (int i = 0; i < 32; ++i) {
////        cout << i << " <> " << ((v2 >> i) & 1) << endl;
////    }
//    return 0;
//    const __m128 t1 = _mm_movehl_ps(v, v);
//    const __m128 t2 = _mm_add_ps(v, t1);
//    const __m128 sum = _mm_add_ss(t1, _mm_shuffle_ps(t2, t2, 1));
//    _mm_storeu_si128((__m128i *)testOut.data(), sum);
//    for (int i = 0; i < testOut.size(); i++) {
//        cout << i << " > " << (int)testOut[i] << endl;
//    }
//    return 0;
    
    const int resizeN = 2;
    cv::Mat img = imread("/Users/alex/Desktop/2022-12-16_14-53-14/snapshot_0.jpg",  IMREAD_GRAYSCALE);
//    cv::Mat img = imread("/Users/alex/Downloads/image_processing20201123-8941-1um70ga.jpg",  IMREAD_GRAYSCALE);
    cv::Mat imgF;
//    img(Rect(0, 0, img.cols-30, img.rows-1)).copyTo(img);
    img.convertTo(imgF, CV_32F, 1/255.0);
    cv::Mat small, smallResize, smallF, smallF2;
    std::vector<uint8_t> small2Vec, small3Vec;
    std::vector<float> small2VecF, small3VecF;
    cv::Mat realResize;
    manualResize(img, realResize, resizeN);
//    cv::setNumThreads(0);
    for (int i = 0; i < 300; i++) {
        R_BENCHMARK("sse_uint8") {
//            small2Vec.resize(img.size().area() /3);
            if (resizeN==2) {
                ssePyrdown2(img.data, img.rows, img.cols, small2Vec);
            } else if (resizeN == 3) {
                ssePyrdown3(img.data, img.rows, img.cols, small2Vec);
            } else {
                ssePyrdown4(img.data, img.rows, img.cols, small2Vec);
            }
        }
        R_BENCHMARK("sse_float") {
            ssePyrdownF((float*)imgF.data, imgF.rows, imgF.cols, small2VecF);
        }
        R_BENCHMARK("cv_pyrdown_float") {
            cv::pyrDown(imgF, smallF2);
        }
        R_BENCHMARK("cv_pyrdown_uint8") {
            cv::pyrDown(img, small);
        }
        R_BENCHMARK("cv_resize_uint8") {
            cv::resize(img, smallResize, img.size() / resizeN, 0, 0, INTER_AREA);
        }
        R_BENCHMARK("eigen_pyrdown_float") {
            eigenPyrdown((float*)imgF.data, imgF.rows, imgF.cols, small3VecF);
        }
        cout << (int)small.data[0] << " <> "
        << (int)small2Vec[0] << " <> "
        << (int)smallResize.data[0] << " || "
        << small2VecF[0] << " <> "
        << smallF2.at<float>(0, 0) << " <> "
        << small3VecF[0] << endl;
//        cout << (int)small.data[0] << " <> " << (int)small2Vec[0] << " <> " << (int)smallResize.data[0] << endl;
//        cout << (int)small.data[0] << " <> " << (int)small2Vec[0] << endl;
    }
    
    
//    for (int i = 0; i < 100; i++) {
//        R_BENCHMARK("cv_f") {
//            cv::pyrDown(imgF, smallF);
//        }
//        R_BENCHMARK("eigen_f") {
//            eigenPyrdown<float>((float*)imgF.data, img.rows, img.cols, small2VecF);
//        }
//        cout << (int)(((float*)smallF.data)[0]) << " <> " << (int)small2Vec[0] << endl;
//    }
    
    cout << R_BENCHMARK_LOG() << endl;
    cv::Mat small2(img.rows/resizeN, img.cols/resizeN, CV_8UC1, small2Vec.data());
    cv::Mat small2f(img.rows/2, img.cols/2, CV_32FC1, small2VecF.data());
    
    cout << "Size 1: " << small.size() << endl;
    cout << "Size 2: " << small2.size() << endl;
    imshow("small_1", small);
    imshow("small_2", small2);
    imshow("small_area", smallResize);
    imshow("small_float", small2f);
    Mat diff;
    double minVal, maxVal;
    Point loc;
    absdiff(small2, realResize, diff);
    cv::minMaxLoc(diff, &minVal, &maxVal, nullptr, &loc);
    cout << maxVal << " " << loc << " >>> " << cv::sum(diff) << endl;
    
    imshow("diff_x10", diff*10);
    waitKey();
    
    return 0;
}
