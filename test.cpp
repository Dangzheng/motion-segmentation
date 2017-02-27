#include <iostream>
#include <math.h>
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>
//
#include <libiomp/omp.h>
// png++
#include "MotionSegmentation.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <png++/png.hpp>

void motionSegmentation(cv::Mat leftImage) {

  std::fstream uFile, vFile;

  int width_ = leftImage.cols;
  int height_ = leftImage.rows;
  float u[width_][height_], v[width_][height_];
  // float u[height_][width_], v[height_][width_];
  uFile.open("u_epic_sintel.txt", std::ios::in);
  vFile.open("v_epic_sintel.txt", std::ios::in);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      uFile >> u[x][y];
      vFile >> v[x][y];
    }
  }
  uFile.close();
  vFile.close();

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> base(width_ * height_,
                                                            3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> match(width_ * height_,
                                                             3);

  base.setOnes();
  match.setOnes();
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (std::isnan(u[x][y]) || std::isnan(v[x][y])) {
        std::cout << "find a nan at (" << x << "," << y << ")" << std::endl;
        std::cout << u[x][y] << std::endl;
        std::cout << v[x][y] << std::endl;
        cv::waitKey(0);
      }
      base(x + y * width_, 0) = static_cast<float>(x); //按行排成两列(x,y)
      base(x + y * width_, 1) = static_cast<float>(y);
      base(x + y * width_, 2) = 1.0;

      match(x + y * width_, 0) =
          base(x + y * width_, 0) + static_cast<float>(u[x][y]);
      match(x + y * width_, 1) =
          base(x + y * width_, 1) + static_cast<float>(v[x][y]);
      // 4位有效数字
      match(x + y * width_, 0) =
          floor(match(x + y * width_, 0) * 10000.000f + 0.5) / 10000.000f;
      match(x + y * width_, 1) =
          floor(match(x + y * width_, 1) * 10000.000f + 0.5) / 10000.000f;
      match(x + y * width_, 2) = 1.0;
    }
  }

  MotionSegmentation segmentation;
  segmentation.setImage(leftImage);
  segmentation.setOptialFlow(base, match);
  segmentation.performMotionSegmentation();
}

int main(int argc, char *argv[]) {
  std::cout << "今晚要上演的是：一幕光荣的救赎..." << std::endl;
  // usage: ./test frame_0020.png
  std::string leftImageFilename = argv[1];
  cv::Mat leftImage = cv::imread(leftImageFilename, CV_LOAD_IMAGE_GRAYSCALE);
  // for (int i = 0; i < 100; i++) {
  //   std::cout << (i + 1) % 5 << std::endl;
  // }
  // cv::waitKey(0);
  motionSegmentation(leftImage);
  return 0;
}
