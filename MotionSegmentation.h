#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <ctime>
#include <vector>

#include <opencv2/opencv.hpp>

#include <png++/png.hpp>

#include <time.h>

class MotionSegment {
public:
  MotionSegment() {}
  void addMatchingPoint(cv::Point2f point1, cv::Point2f point2) {
    matchPoints1.push_back(point1);
    matchPoints2.push_back(point2);
  }

  bool havePoints() {
    bool havePoints = true;
    if (matchPoints1.size() == 0) {
      havePoints = false;
    }
    return havePoints;
  }

  void calcRotationVector() {
    if (matchPoints1.size() < 8) {
      throw std::invalid_argument("[SPSFlow::calcRotationVector] "
                                  "matchPoints's points number less than 8...");
    }

    std::vector<uchar> inliers(matchPoints1.size(), 0);
    cv::Mat fundamental = cv::findFundamentalMat(
        matchPoints1, matchPoints2, inliers, CV_FM_RANSAC, 3, 0.99);

    // int index = 0;
    // int removePixels = 0;
    // for (; itIn != inliers.end(); ++itIn, ++index) {
    //   if (*itIn == false) {
    //     std::vector<cv::Point2f>::const_iterator itM1 = matchPoints1.begin();
    //     std::vector<cv::Point2f>::const_iterator itM2 = matchPoints2.begin();
    //     std::advance(itM1, index - removePixels);
    //     std::advance(itM2, index - removePixels);
    //     matchPoints1.erase(itM1);
    //     matchPoints2.erase(itM2);
    //     ++removePixels;
    //   }
    // }
    //
    std::vector<uchar>::const_iterator itIn = inliers.begin();
    std::vector<cv::Point2f>::const_iterator itM1 = matchPoints1.begin();
    std::vector<cv::Point2f>::const_iterator itM2 = matchPoints2.begin();
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (; itIn != inliers.end(); ++itIn, ++itM1, ++itM2) {
      if (*itIn == true) {
        points1.push_back(*itM1);
        points2.push_back(*itM2);
      }
    }
    //   std::cout << "matching points:" << matchPoints1.size() << std::endl;
    if (matchPoints1.size() < 8) {
      throw std::invalid_argument(
          "[SPSFlow::calcRotationVector] "
          "matchPoints's points after RANSAC number less than 8...");
    }

    //   std::cout << "removed points:" << removePixels << std::endl;
    F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT, 3, 0.99);
    cv::Mat R;
    cv::Mat K(3, 3, CV_64F);
    cv::Mat W(3, 3, CV_64F);
    K = (cv::Mat_<double>(3, 3) << 721.537700, 0.000000, 609.559300, 0.000000,
         721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
    cv::SVD svd(K.t() * F * K, cv::SVD::MODIFY_A);
    W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    R = svd.u * W * svd.vt;
    //   std::cout << "Fundamental:" << std::endl << F << std::endl;
    cv::Mat vector_Rot;
    cv::Rodrigues(R, vector_Rot);
    rotationVector_[0] = vector_Rot.at<double>(0, 0);
    rotationVector_[1] = vector_Rot.at<double>(0, 1);
    rotationVector_[2] = vector_Rot.at<double>(0, 2);
    //   std::cout << "Rotation:" << vector_Rot << std::endl;
  }

  std::vector<cv::Point2f> getPoints() { return matchPoints1; }

  std::vector<cv::Point2f> getMatchPoints() { return matchPoints2; }

  void clear() {
    matchPoints1.clear();
    matchPoints2.clear();
    rotationVector_[0] = -1;
    rotationVector_[1] = -1;
    rotationVector_[2] = -1;
  }

  double parameter(int index) { return rotationVector_[index]; }

  void appendSegment(std::vector<cv::Point2f> pt1,
                     std::vector<cv::Point2f> pt2) {
    matchPoints1.insert(matchPoints1.end(), pt1.begin(), pt1.end());
    matchPoints2.insert(matchPoints2.end(), pt2.begin(), pt2.end());
  }

  void copyOldVersion() {
    oldVersion_[0] = rotationVector_[0];
    oldVersion_[1] = rotationVector_[1];
    oldVersion_[2] = rotationVector_[2];
  }

  bool isStable() {
    if (oldVersion_[0] == rotationVector_[0] &&
        oldVersion_[1] == rotationVector_[1] &&
        oldVersion_[2] == rotationVector_[2]) {
      return true;
    } else {
      return false;
    }
  }

  cv::Mat getFundamentalMatrix() { return F; }

  // void calcSymetricDistance(cv::Mat fundamentalMatrix) {
  //   std::vector<cv::Point2f>::const_iterator itM1 = matchPoints1.begin();
  //   std::vector<cv::Point2f>::const_iterator itM2 = matchPoints2.begin();
  //   cv::Mat forward, backword;
  //   float d;
  //   for (; itM1 < matchPoints1.begin(); ++itM1, ++itM2) {
  //     cv::Mat l(3, 1, CV_32F, 1);
  //     cv::Mat r(3, 1, CV_32F, 1);
  //     l.at<float>(0, 0) = itM1->x;
  //     l.at<float>(1, 0) = itM1->y;
  //     r.at<float>(0, 0) = itM2->x;
  //     r.at<float>(1, 0) = itM2->y;
  //     forward = l.t() * fundamentalMatrix * r;
  //     backword = r.t() * fundamentalMatrix.t() * l;
  //     d = pow(forward.at<float>(0, 0), 2) + pow(backword.at<float>(0, 0), 2);
  //     symetricDistanceCost_(int(itM1->x), int(itM1->y)) = d;
  //   }
  // }

  // Eigen::MatrixXf getSymetricCostImage() { return symetricDistanceCost_; }

private:
  double rotationVector_[3];
  double oldVersion_[3];
  std::vector<cv::Point2f> matchPoints1;
  std::vector<cv::Point2f> matchPoints2;
  // need initial or not?
  cv::Mat F;
  Eigen::MatrixXf symetricDistanceCost_;
};
//-------------------------------------MotionSegmentation-----------------------------------
class MotionSegmentation {
public:
  MotionSegmentation() {}

  void performMotionSegmentation();
  void makeDerivatives2D();
  // void setImage(const png::image<png::rgb_pixel> &leftImage);
  void setImage(cv::Mat leftImage);
  void setOptialFlow(Eigen::MatrixXf base, Eigen::MatrixXf match);
  Eigen::MatrixXf calcSymetricDistance(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          fundamentalMatrix);

private:
  void performPrimalDual(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          &symCost,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u,
      std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>> &motionProposals);
  void forwardBackwardConsistencyCheck(float threshold);
  // int isMember(int x);
  void updateFundamentalMatrix(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u,
      std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>> &labelFundamentalMat);
  void updateSymmetryCost(
      std::vector<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          labelFundamentalMat,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          &symCost);
  bool checkEnergyDescending(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          symCost,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u,

      float &energyLastTime, float lambda);
  void generateMotionProposals(
      std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>> &motionProposals);
  void recoverHardAssignment(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u);

  void modelDiscovery(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u,
      std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>> &motionProposals);
  cv::Mat ccaByTwoPass(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          outlier);
  // void modelClustering(
  //     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //     &u,
  //     std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  //                               Eigen::RowMajor>> &motionProposals);
  // define grad and div, 将图片按照行展开，排成列向量。
  int width_, height_;
  Eigen::SparseMatrix<float> D, Dx, Dy, div;
  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Wl;
  // float *symetricDistanceImage_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> baseVec_, matchVec_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> outliers_;
  std::vector<cv::Point2f> points1, points2;
};

//-----------------------------------Kmeans-----------------------------------

class KMeans {
public:
  KMeans();
  void setK(const int k);
  void setIterateMax(const int iterMax);
  std::vector<MotionSegment> compute(std::vector<MotionSegment> motionSegments);

private:
  int k_;
  int iterMax_;
  double random(double start, double end);
  double calcAngle(double a1, double a2, double a3, double b1, double b2,
                   double b3);
};
