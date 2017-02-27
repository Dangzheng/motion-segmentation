
#include "MotionSegmentation.h"
#include "defParameters.h"
#include "opencv2/core/eigen.hpp"
const int KMEANS_DEFAULT_K = 7;
const int ITERATE_MAX_DEFAULT_NUMBER = 20;

KMeans::KMeans() : k_(KMEANS_DEFAULT_K), iterMax_(ITERATE_MAX_DEFAULT_NUMBER) {}

void KMeans::setK(const int k) {
  if (k < 2) {
    throw std::invalid_argument("[KMeans::setK] "
                                "K factor is less than 2");
  }
  k_ = k;
}

void KMeans::setIterateMax(const int iterMax) {
  if (iterMax < 1) {
    throw std::invalid_argument("[KMeans::setK] "
                                "Max iterations number is less than 1");
  }
  iterMax_ = iterMax;
}

std::vector<MotionSegment>
KMeans::compute(std::vector<MotionSegment> motionSegments) {
  //首先，随机选取数个Superpixel，然后作为初始的均值向量。
  bool convergence = false;
  int iter = 0;
  std::vector<MotionSegment> cluster(k_);
  while (convergence == false && iter < iterMax_) {
    srand(unsigned(time(0)));
    for (std::vector<MotionSegment>::iterator it = cluster.begin();
         it < cluster.end(); ++it) {
      it->clear();
      std::cout << "random:" << int(random(0, motionSegments.size()))
                << std::endl;
      *it = motionSegments[int(random(0, motionSegments.size()))];
    }
    cv::waitKey(0);
    for (std::vector<MotionSegment>::iterator itM = motionSegments.begin();
         itM < motionSegments.end(); ++itM) {
      std::vector<double> angle(k_);
      std::vector<MotionSegment>::iterator itC = cluster.begin();
      std::vector<double>::iterator itA = angle.begin();
      for (; itC < cluster.end(); ++itC, ++itA) {
        *itA =
            calcAngle(itC->parameter(0), itC->parameter(1), itC->parameter(2),
                      itM->parameter(0), itM->parameter(1), itM->parameter(2));
      }
      std::vector<double>::iterator min =
          std::min_element(std::begin(angle), std::end(angle));
      //在这个位置Cluster直接接收segment中的所有的点。
      itC = cluster.begin();
      std::advance(itC, std::distance(std::begin(angle), min));
      itC->appendSegment(itM->getPoints(), itM->getMatchPoints());
    }
    convergence = true;
    for (std::vector<MotionSegment>::iterator it = cluster.begin();
         it < cluster.end(); ++it) {
      it->calcRotationVector();
      if (it->isStable() == false) {
        convergence = false;
      }
    }
  }
  std::cout << "I'm here~~~" << std::endl;
  cv::waitKey(0);
  return cluster;
}

double KMeans::random(double start, double end) {
  return start + (end - start) * rand() / (RAND_MAX + 1.0);
}

double KMeans::calcAngle(double a1, double a2, double a3, double b1, double b2,
                         double b3) {
  double angle =
      (a1 * b1 + a2 * b2 + a3 * b3) /
      (sqrt(a1 * a1 + a2 * a2 + a3 * a3) * sqrt(b1 * b1 + b2 * b2 + b3 * b3));
  return angle;
}

// MotionSegmentation;

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X,
              cv::Mat &Y) {
  // usage:
  // cv::Mat x, y;
  // meshgrid(cv::Range(0, width_ - 1), cv::Range(0, height_ - 1), x, y);
  std::vector<float> t_x, t_y;
  for (int i = xgv.start; i <= xgv.end; i++)
    t_x.push_back(i);
  for (int j = ygv.start; j <= ygv.end; j++)
    t_y.push_back(j);

  cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X); //写好了一列，重复y行
  cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);     //写好了一行重复x列。
}

cv::Mat mergeRows(cv::Mat A, cv::Mat B) {
  // cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
  int totalRows = A.rows + B.rows;
  cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
  cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
  A.copyTo(submat);
  submat = mergedDescriptors.rowRange(A.rows, totalRows);
  B.copyTo(submat);
  return mergedDescriptors;
}

cv::Mat mergeCols(cv::Mat A, cv::Mat B) {
  // cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
  int totalCols = A.cols + B.cols;
  cv::Mat mergedDescriptors(A.rows, totalCols, A.type());
  cv::Mat submat = mergedDescriptors.colRange(0, A.cols);
  A.copyTo(submat);
  submat = mergedDescriptors.colRange(A.cols, totalCols);
  B.copyTo(submat);
  return mergedDescriptors;
}

void MotionSegmentation::makeDerivatives2D() {
  int length = width_ * height_; // height_:375
  std::vector<Eigen::Triplet<float>> tripletList;
  std::vector<Eigen::Triplet<float>> tripletTotal;
  Eigen::SparseMatrix<float> mat(length * 2, length);
  Eigen::SparseMatrix<float> matx(length, length);
  Eigen::SparseMatrix<float> maty(length, length);

  cv::Mat linIdx(height_, width_, CV_32F);
  cv::Mat rowIdx, colIdx, mergeUp, mergeBot, scalar;
  cv::Mat val = cv::Mat(length * 2, 1, CV_32F, 1);

  cv::Mat_<float>::iterator itL = linIdx.begin<float>();
  int index = 0;
  for (; itL < linIdx.end<float>(); ++itL, ++index) {
    *itL = float(index);
    //因为opencv是从零开始计数的，因为这个也是作为索引值出现，所以此处也是从零开始计数。
  }

  mergeUp = mergeCols(linIdx.colRange(0, width_ - 1).clone().reshape(0, 1),
                      linIdx.colRange(0, width_ - 1).clone().reshape(0, 1));
  mergeBot =
      mergeRows(linIdx.col(width_ - 1).clone(), linIdx.col(width_ - 1).clone());
  rowIdx = mergeRows(mergeUp.t(), mergeBot);

  mergeUp = mergeCols(linIdx.colRange(0, width_ - 1).clone().reshape(0, 1),
                      linIdx.colRange(1, width_).clone().reshape(0, 1));
  mergeBot =
      mergeRows(linIdx.col(width_ - 1).clone(), linIdx.col(width_ - 2).clone());
  colIdx = mergeRows(mergeUp.t(), mergeBot);

  val.rowRange(0, height_ * (width_ - 1)).setTo(-1);
  val.rowRange(val.rows - height_, val.rows).setTo(-1);

  cv::Mat_<float>::iterator itR = rowIdx.begin<float>();
  cv::Mat_<float>::iterator itC = colIdx.begin<float>();
  cv::Mat_<float>::iterator itV = val.begin<float>();

  for (; itV < val.end<float>(); ++itR, ++itC, ++itV) {
    tripletList.push_back(Eigen::Triplet<float>(*itR, *itC, *itV));
    tripletTotal.push_back(Eigen::Triplet<float>(*itR, *itC, *itV));
  }
  matx.setFromTriplets(tripletList.begin(), tripletList.end());

  mergeUp = mergeCols(linIdx.rowRange(0, height_ - 1).clone().reshape(0, 1),
                      linIdx.rowRange(0, height_ - 1).clone().reshape(0, 1));
  mergeBot = mergeCols(linIdx.row(height_ - 1).clone(),
                       linIdx.row(height_ - 1).clone());
  rowIdx = mergeRows(mergeUp.t(), mergeBot.t());

  mergeUp = mergeCols(linIdx.rowRange(0, height_ - 1).clone().reshape(0, 1),
                      linIdx.rowRange(1, height_).clone().reshape(0, 1));
  mergeBot = mergeCols(linIdx.row(height_ - 1).clone(),
                       linIdx.row(height_ - 2).clone());
  colIdx = mergeRows(mergeUp.t(), mergeBot.t());

  val = cv::Mat::ones(length * 2, 1, CV_32F);
  val.rowRange(0, (height_ - 1) * width_).setTo(-1);
  val.rowRange(val.rows - width_, val.rows).setTo(-1);
  itR = rowIdx.begin<float>();
  itC = colIdx.begin<float>();
  itV = val.begin<float>();
  tripletList.clear();
  for (; itV < val.end<float>(); ++itR, ++itC, ++itV) {
    tripletList.push_back(Eigen::Triplet<float>(*itR, *itC, *itV));
    tripletTotal.push_back(Eigen::Triplet<float>(*itR + length, *itC, *itV));
  }

  mat.setFromTriplets(tripletTotal.begin(), tripletTotal.end());
  maty.setFromTriplets(tripletList.begin(), tripletList.end());

  D = Eigen::SparseMatrix<float>(mat);
  Dx = Eigen::SparseMatrix<float>(matx);
  Dy = Eigen::SparseMatrix<float>(maty);
  div = D.transpose();
  // cv::Mat x, y;
  // meshgrid(cv::Range(0, width_ - 1), cv::Range(0, height_ - 1), x, y);
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> test(
  //     height_, width_);
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  // truth;
  // cv2eigen(x, test);
  // test.resize(width_ * height_, 1);
  // std::cout << test.block(0, 0, 10, 1) << std::endl;
  // truth = Dy * test;
  // std::cout << truth.block(0, 0, 10, 1) << std::endl;
  // truth.resize(height_, width_);
  // cv::Mat w;
  // eigen2cv(truth, w);
  // cv::imwrite("result.png", w);
  // cv::waitKey(0);
  // std::cout << *itR << ":" << *itC << std::endl;
  // std::cout << "D:" << std::endl << Dx.block(0, 0, 10, 10) << std::endl;
  // cv::waitKey(0);
}

void MotionSegmentation::setImage(cv::Mat leftImage) {
  width_ = leftImage.cols;
  height_ = leftImage.rows;
  // make_derivatives_2D_complete
  makeDerivatives2D();
  // cv::imwrite("gray.png", leftImage);
  leftImage.convertTo(leftImage, CV_32FC1);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      leftVecRow(leftImage.rows, leftImage.cols);
  cv2eigen(leftImage, leftVecRow);
  leftVecRow = leftVecRow / 255;
  // std::cout << leftVecRow.block(0, 0, 10, 10) << std::endl;
  leftVecRow.resize(width_ * height_, 1);
  // std::cout << leftVecRow.block(0, 0, 10, 1) << std::endl;
  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dIx, dIy,
      norm;
  dIx = Dx * leftVecRow; // dx,dy 与作者的计算的结果不同。
  dIy = Dy * leftVecRow;

  //
  // std::vector<Eigen::Triplet<float>> triplets;
  // for (int i = 0; i < width_ * height_; ++i) {
  //   float norm = pow(dIx(i, 0), 2) + pow(dIy(i, 0), 2);
  //   // float norm = pow(grad(i, 0), 2);
  //   norm = -1 * beta * norm;
  //   float diagValue = exp(norm);
  //   triplets.push_back(Eigen::Triplet<float>(i, i, diagValue));
  // }
  // Eigen::SparseMatrix<float, Eigen::RowMajor> mat(2 * width_ * height_,
  //                                                 2 * width_ * height_);
  // mat.setFromTriplets(triplets.begin(), triplets.end());
  // Wl = Eigen::SparseMatrix<float>(mat);
  norm = -1 * beta * (dIx.square() + dIy.square());
  Wl = norm.exp();
  // cv::waitKey(0);
}

void MotionSegmentation::setOptialFlow(Eigen::MatrixXf base,
                                       Eigen::MatrixXf match) {
  //此处还缺少error function检测输入是否为空；
  baseVec_ = base;
  matchVec_ = match;
};

cv::Scalar getRandomColor(int pixelValue) {
  uchar r = 255 * (rand() / (1.0 + RAND_MAX));
  uchar g = 255 * (rand() / (1.0 + RAND_MAX));
  uchar b = 255 * (rand() / (1.0 + RAND_MAX));
  // uchar r = 255 / 9 * (static_cast<float>(pixelValue));
  // uchar g = 255 / 9 * (static_cast<float>(pixelValue));
  // uchar b = 255 / 9 * (static_cast<float>(pixelValue));
  return cv::Scalar(b, g, r);
}

void labelColor(const cv::Mat &_labelImg, cv::Mat &_colorLabelImg) {
  if (_labelImg.empty() || _labelImg.type() != CV_32SC1) {
    return;
  }

  std::map<int, cv::Scalar> colors;

  int rows = _labelImg.rows;
  int cols = _labelImg.cols;

  _colorLabelImg.release();
  _colorLabelImg.create(rows, cols, CV_8UC3);
  _colorLabelImg = cv::Scalar::all(0);

  for (int i = 0; i < rows; i++) {
    const int *data_src = (int *)_labelImg.ptr<int>(i);
    uchar *data_dst = _colorLabelImg.ptr<uchar>(i);
    for (int j = 0; j < cols; j++) {
      int pixelValue = data_src[j];
      if (pixelValue > 0) {
        if (colors.count(pixelValue) <= 0) {
          colors[pixelValue] = getRandomColor(pixelValue);
        }
        cv::Scalar color = colors[pixelValue];
        *data_dst++ = color[0];
        *data_dst++ = color[1];
        *data_dst++ = color[2];
        // 在这里赋给像素点RGB值，颜色全部都是从getRandomColor里面取来的
        // 如果我给randomcolor锁定一个值，让他的颜色固定，就好办了。
        //
      } else {
        data_dst++;
        data_dst++;
        data_dst++;
      }
    }
  }
}

int writeData(std::string fileName, cv::Mat &matData) {
  int retVal = 0;

  // 检查矩阵是否为空
  if (matData.empty()) {
    std::cout << "the Matrix is empty..." << std::endl;
    retVal = 1;
    return (retVal);
  }

  // 打开文件
  std::ofstream outFile(fileName.c_str(),
                        std::ios_base::out); //按新建或覆盖方式写入
  if (!outFile.is_open()) {
    std::cout << "faile in open file..." << std::endl;
    retVal = -1;
    return (retVal);
  }

  // 写入数据
  for (int r = 0; r < matData.rows; r++) {
    for (int c = 0; c < matData.cols; c++) {
      // int data = matData.at<uchar>(
      //     r, c); //读取数据，at<type> - type 是矩阵元素的具体数据格式
      //     uchar
      float data = matData.at<float>(r, c);
      outFile << data << "\t"; //每列数据用 tab 隔开
    }
    outFile << std::endl; //换行
  }

  return (retVal);
}

void MotionSegmentation::performMotionSegmentation() {
  Eigen::setNbThreads(8);
  forwardBackwardConsistencyCheck(checkThreshold);
  std::vector<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      motionProposals, oldMotionProposalSet;
  generateMotionProposals(motionProposals);
  oldMotionProposalSet = motionProposals;
  // get motion proposal(fundamental matrix) for motion segmentation
  // calculating the symCost at the same time
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> symCost(
      width_ * height_, 1);
  // const float gamma = 0.0010; //给outlier的固定cost值。
  const float gamma = 1;
  symCost.col(symCost.cols() - 1).setConstant(gamma);
  symCost.col(0) = (outliers_.array() == 1).select(0.0, symCost.col(0));
  // outlier有一个原则：一朝为异常，永生为异常，再无翻身。
  // outlier 标签放在第一列上，这样以后好分辨。
  for (std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>::iterator it =
           motionProposals.begin() + 1;
       it < motionProposals.end(); ++it) {
    symCost.conservativeResize(Eigen::NoChange, symCost.cols() + 1);
    symCost.col(symCost.cols() - 1) = calcSymetricDistance(*it);
  }
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //     background;
  // cv::Mat show;
  // background = symCost.col(1);
  // background.resize(height_, width_);
  // cv::eigen2cv(background, show);
  // writeData("symCost.txt", show);
  // std::cout << "data has been writen" << std::endl;
  // cv::waitKey(0);

  // std::cout << "SymCost:" << symCost.block(0, 0, 100, symCost.cols());
  // cv::waitKey(0);
  std::cout << "#initial symmetric cost image finish..." << std::endl;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u(
      symCost.rows(), symCost.cols());
  u.setConstant(1.0 / u.cols());
  // int i;
  // #pragma omp parallel for private(i)
  //   for (i = 0; i < u.rows(); ++i) {
  //     if (outliers_(i, 0) == 1) {
  //       u.row(i).setZero();
  //       u(i, 0) = 1.0;
  //     }
  //   }
  // 作者自己说的不超过10步就能
  for (int iter = 0; iter < 3; ++iter) {
    performPrimalDual(symCost, u, motionProposals);
    std::cout << "#primal-dual one round finish..." << std::endl;
    recoverHardAssignment(u); //显示一下Motion segmentation结果
    modelDiscovery(u, motionProposals);
    // 就在这个位置加一个聚类的过程，这样就好了
    // modelClustering(u, motionProposals);
    std::cout << "new model discovery finish..." << std::endl;
    updateSymmetryCost(motionProposals, symCost);
  }
}

Eigen::MatrixXf MotionSegmentation::calcSymetricDistance(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        fundamentalMatrix) {
  //所谓的对称距离就是点到极线的距离，就直接用点到直线距离公式就好了。
  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> epilineB,
      epilineM, foreward, backward, foreBot, backBot, foreTop, backTop;
  // denormalize
  epilineB = fundamentalMatrix.transpose() * matchVec_.transpose();
  epilineM = fundamentalMatrix * baseVec_.transpose();
  // epilineB = fundamentalMatrix.transpose() * matchVec_.transpose();
  // epilineM = fundamentalMatrix * baseVec_.transpose();
  foreward = baseVec_.array() * epilineB.transpose();
  backward = matchVec_.array() * epilineM.transpose();
  // Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //     baseNorVec, matchNorVec;
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N(3,
  // 3);
  // N << 2.0 / width_, 0.0, -1.0, 0.0, 2.0 / height_, -1.0, 0.0, 0.0, 1.0;
  // baseNorVec = N * baseVec_.transpose();
  // matchNorVec = N * matchVec_.transpose();
  //
  // // normalized fundamental1 matrix
  // fundamentalMatrix = N.transpose().inverse() * fundamentalMatrix *
  // N.inverse();
  // // normalize
  // epilineB = fundamentalMatrix * matchNorVec.matrix();
  // epilineM = fundamentalMatrix.transpose() * baseNorVec.matrix();
  // foreward = baseNorVec.transpose() * epilineB.transpose();
  // backward = matchNorVec.transpose() * epilineM.transpose();

  foreTop = foreward.rowwise().sum().square();
  backTop = backward.rowwise().sum().square();
  // A^2+B^2,divid
  foreBot = epilineB.row(0).square() + epilineB.row(1).square();
  backBot = epilineM.row(0).square() + epilineM.row(1).square();

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      costVec = foreTop / foreBot.transpose() + backTop / backBot.transpose();
  // float scale = 0.001;
  // costVec = costVec * scale;
  costVec = (costVec.array() > 1).select(1.0, costVec);
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cost;
  // cost = costVec;
  // cost.resize(436, 1024);
  // cv::Mat w(436, 1024, CV_32FC1);
  // eigen2cv(cost, w);
  // // w.convertTo(w, CV_32SC1);
  // writeData("result.txt", w);
  // std::cout << "data has been writen..." << std::endl;
  // cv::waitKey(10000);
  return costVec;
}

// float aboveOne(float x) {
//   if (x < 1)
//     return +1.0;
//   else
//     return x;
// }

void MotionSegmentation::performPrimalDual(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        &symCost,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u,
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>> &motionProposals) {
  // initial；
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_;
  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lc;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
      2 * symCost.rows(), symCost.cols());

  u_ = u;
  p.setZero();
  float L2 = 8.0;
  float tau = 1.0 / sqrt(L2);
  float sigma = 1.0 / sqrt(L2);
  float K = u.cols();
  float lambda = 1;

  lc = -2 * tau * lambda * symCost;
  lc = lc.exp();
  bool descending = true;
  bool converged = false;
  float energy = FLT_MAX;
  int maxIter = 2000;
  int iterCt = 0;

  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_sum(u);
  u_sum.setZero();
  while (converged == false) {
    for (int inner = 0; inner < maxIter; inner++) {
      // 1.dual update
      // p += Wl * sigma * (D * u_);
      p += sigma * (D * u_);
      // reproject dual variables
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> norm,
          norm_;
      bool isotropic = true;
      if (isotropic == true) {
        norm = p.array().topRows(width_ * height_).square() +
               p.array().bottomRows(width_ * height_).square();
        norm_.conservativeResize(norm.rows() * 2, norm.cols());
        norm = norm.sqrt() / Wl.replicate(1, K);
        // std::cout << "hello" << std::endl;
        norm_ = norm.replicate(2, 1);
        norm_ = (norm_ < 1).select(1.0, norm_);
        // norm_ = norm.array().unaryExpr(std::ptr_fun(aboveOne));
        p = p.array() / norm_;
      } else {
        norm = p.array().abs();
        norm = (p.array() < 1).select(1.0, norm);
        p = p.array() / norm.array();
      }

      // 2.primal update
      bool ergodic = true;

      u_ = u; // remember old u
      if (ergodic == false) {
        // float constant = 2.4;
        // u -= tau * (-lambda * div * Wl * p + constant / lambda * symCost);
        u -= tau * (symCost * lambda - div * p);

        // this should use sparse matrix
        // reproject primal variable onto simplex
        Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mu(
            u.rows(), 1);
        mu.setConstant(FLT_MIN);
        // u的列数就是Motion proposal的个数。
        // reproject primal variable onto simplex
        for (int i = 0; i < u.cols(); i++) {
          Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
              a(u.rows(), 1);
          Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
              b(u.rows(), 1);
          a.setZero();
          b.setZero();
          int j;
          Eigen::initParallel();
          // #pragma omp parallel for private(j)
          for (j = 0; j < u.cols(); j++) {
            Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                tmp;
            tmp = u.col(j);
            a = (tmp > mu).select(a + tmp, a);
            b = (tmp > mu).select(b + 1.0, b);
          }
          mu = (a - 1) / b;
        }
        Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            mu_;
        mu_ = mu.replicate(1, u.cols());
        u = u.array() - mu_;
        u = (u.array() < 0).select(0, u);
      } else {
        Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            tmp, tmp1, tmp2;
        // better calculating on GPU
        tmp = -2 * tau * div * p;
        tmp1 = tmp.exp() * lc * u.array();
        u = tmp1 / (tmp1.rowwise().sum().replicate(1, u.cols()));
        //这个位置这么写直接就避免了上一种欧式setting中必须要计算单纯型限制的步骤。
        //提速巨大。
      }

      u_ = 2 * u.array() - u_.array();
      u_sum = (inner) / (inner + 1) * u_sum + u.array() / (inner + 1);
      std::cout << "inner: " << inner << std::endl;
      if ((inner + 1) % 50 == 0) {
        descending = checkEnergyDescending(symCost, u, energy, lambda);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            show;
        show = u_sum;
        recoverHardAssignment(show);
        if (descending == false) {
          break;
        }
      }
    }
    updateFundamentalMatrix(u_sum, motionProposals);
    if (iterCt > 3) {
      converged = true;
    }
    iterCt++;
  }
}

void MotionSegmentation::forwardBackwardConsistencyCheck(float threshold) {
  //首先载入backward的optical flow���������果�������
  std::fstream uFile, vFile;
  uFile.open("u_epic_sintel_back.txt", std::ios::in);
  vFile.open("v_epic_sintel_back.txt", std::ios::in);
  float u[width_][height_], v[width_][height_];
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      uFile >> u[x][y];
      vFile >> v[x][y];
    }
  }
  uFile.close();
  vFile.close();
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> backward(width_ * height_,
                                                              2);
  backward.setOnes();
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (std::isnan(u[x][y]) || std::isnan(v[x][y])) {
        std::cout << "find a nan at (" << x << "," << y << ")" << std::endl;
        std::cout << u[x][y] << std::endl;
        std::cout << v[x][y] << std::endl;
        cv::waitKey(0);
      }
      backward(x + y * width_, 0) = std::floor(x + u[x][y]);
      backward(x + y * width_, 1) = std::floor(y + v[x][y]);
    }
  }
  // cv::Mat baseImage = cv::imread("frame_0020.png", 0);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outlier(
      width_ * height_, 1);
  outlier.setZero();
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      // int piForward, piBackward;
      int xMatch, yMatch, xBase, yBase;

      // piForward = (int)baseImage.at<uchar>(y, x);
      xMatch = std::floor(matchVec_(x + y * width_, 0));
      yMatch = std::floor(matchVec_(x + y * width_, 1));
      if (xMatch < 0 || xMatch > width_ - 1 || yMatch < 0 ||
          yMatch > height_ - 1 || backward(xMatch + yMatch * width_, 0) < 0 ||
          backward(xMatch + yMatch * width_, 0) > width_ - 1 ||
          backward(xMatch + yMatch * width_, 1) < 0 ||
          backward(xMatch + yMatch * width_, 1) > height_ - 1) {
        outlier(x + y * width_, 0) = 1.0;
      } else {
        xBase = backward(xMatch + yMatch * width_, 0);
        yBase = backward(xMatch + yMatch * width_, 1);
        // piBackward = (int)baseImage.at<uchar>(yBase, xBase);

        // if (std::abs(piForward - piBackward) > threshold) {
        //   outlier(x + y * width_, 0) = 1.0;
        // }
        if (std::sqrt((x - xBase) * (x - xBase) + (y - yBase) * (y - yBase)) >
            threshold) {
          outlier(x + y * width_, 0) = 1.0;
        }
      }
    }
  }
  // cv::Mat result;
  // outlier.resize(height_, width_);
  // cv::eigen2cv(outlier, result);
  // cv::imshow("outlier", result);
  // cv::waitKey(0);
  outliers_ = outlier;
  std::cout << "outlier's number:" << outlier.sum() << std::endl;
}

void MotionSegmentation::updateFundamentalMatrix(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u,
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>> &motionProposals) {
  for (int i = 0; i < u.rows(); ++i) {
    if (outliers_(i, 0) == 1) {
      // u.row(i).setZero();
      // u(i, 0) = 1.0;
      u.row(i).setZero(); //改
    }
  }
  double clockBegin, clockEnd;
  double omp_get_wtime(void);
  clockBegin = omp_get_wtime();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> match1,
      match2;
  //将坐标统一成齐次坐标
  match1 = baseVec_;
  // match1.conservativeResize(Eigen::NoChange, match1.cols() + 1);
  // match1.col(3).setConstant(1.0);
  match2 = matchVec_;
  // match2.conservativeResize(Eigen::NoChange, match2.cols() + 1);
  // match2.col(3).setConstant(1.0);

  // the matrix for normalization(Centroid)
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N(3, 3);
  N << 2.0 / width_, 0.0, -1.0, 0.0, 2.0 / height_, -1.0, 0.0, 0.0, 1.0;

  // Data Centroid
  match1 = match1 * N.transpose();
  match2 = match2 * N.transpose();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      match1.rows(), 9);
  // set A
  A.col(0) = match1.array().col(0) * match2.array().col(0);
  A.col(1) = match1.array().col(1) * match2.array().col(0);
  A.col(2) = match2.array().col(0);

  A.col(3) = match1.array().col(0) * match2.array().col(1);
  A.col(4) = match1.array().col(1) * match2.array().col(1);
  A.col(5) = match2.array().col(1);

  A.col(6) = match1.array().col(0) * match2.array().col(2);
  A.col(7) = match1.array().col(1) * match2.array().col(2);
  A.col(8) = match2.array().col(2);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_(9,
                                                                           9);
  // std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  //                           Eigen::RowMajor>>::iterator it =
  //     motionProposals.begin() + 1;
  //   更新motion
  //   proposal的时候同���要注意不要把第一个outlier的占位更�����������了。
  // ul���从第二���开���进行匹配。
  int i;
#pragma omp parallel for private(i)
  for (i = 1; i < motionProposals.size(); ++i) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ul;
    ul = u.col(i);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp;
    tmp = ul.array().replicate(1, 9) * A.array();
    tmp = (tmp.array() == -0.0).select(0, tmp);
    A_.setZero();
    A_ = tmp.transpose() * A;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(
        A_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V, F;
    V = svd.matrixV().col(8);
    V.resize(3, 3);
    // make rank 2
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_(
        V, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf diag = svd_.singularValues().asDiagonal();
    diag(2, 2) = 0.0;
    F = svd_.matrixU() * diag * svd_.matrixV().transpose();
    // denormalize
    F = N.transpose() * F * N;

    // std::cout << "F_before" << std::endl << *it << std::endl;
    // std::cout << "F_update" << std::endl << F << std::endl;

    // cv::Mat R;
    // cv::Mat K(3, 3, CV_64F);
    // cv::Mat W(3, 3, CV_64F);
    // K = (cv::Mat_<double>(3, 3) << 721.537700, 0.000000, 609.559300,
    // 0.000000,
    //      721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
    // cv::Mat fundamental;
    // cv::eigen2cv(F, fundamental);
    // fundamental.convertTo(fundamental, CV_64F);
    // cv::SVD opensvd(K.t() * fundamental * K, cv::SVD::MODIFY_A);
    // W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    // R = opensvd.u * W * opensvd.vt;
    // std::cout << "R:" << R << std::endl;
    // std::cout << "========================================" << std::endl;

    motionProposals[i] = F;
    //将更新后的F记录在motionProposalsrix中用于更新symmetryCost
  }
  clockEnd = omp_get_wtime();
  std::cout << "update fundamental matrix cost:"
            << (clockEnd - clockBegin) / CLOCKS_PER_SEC << "sec" << std::endl;
}

void MotionSegmentation::updateSymmetryCost(
    std::vector<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        motionProposals,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        &symCost) {
  double clockBegin, clockEnd;
  double omp_get_wtime(void);
  clockBegin = omp_get_wtime();
  //该有多少F矩阵就更新多少,第一列为outlier的cost不更新。
  symCost.conservativeResize(Eigen::NoChange, 1);
  symCost.conservativeResize(Eigen::NoChange, motionProposals.size());
  // std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  //                           Eigen::RowMajor>>::iterator itLF =
  //     motionProposals.begin() + 1;

  // 每次都要将outlier的symCost更新空过去，因为这个cost是手动给定了的。
  int i;
#pragma omp parallel for private(i)
  for (i = 1; i < motionProposals.size(); ++i) {
    symCost.col(i) = calcSymetricDistance(motionProposals[i]);
  }
  clockEnd = omp_get_wtime();
  std::cout << "update symCost cost:"
            << (clockEnd - clockBegin) / CLOCKS_PER_SEC << "sec" << std::endl;
}

bool MotionSegmentation::checkEnergyDescending(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        symCost,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u,
    float &energyLastTime, float lambda) {
  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dataTerm,
      lengthTerm;
  dataTerm = u.array() * symCost.array();
  lengthTerm = D * u;
  float norm2_1 = 0.0;
  for (int i = 0; i < u.cols(); ++i) {
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp;
    tmp = lengthTerm.col(i);
    tmp.resize(height_, width_);
    tmp.square();
    for (int j = 0; j < tmp.rows(); ++j) {
      norm2_1 += std::sqrt(tmp.row(j).sum());
    }
  }
  float energy = lambda * dataTerm.sum() + norm2_1;
  std::cout << "energy:" << energy << std::endl;
  if (std::abs(energy - energyLastTime) < 0.01) {
    return false;
  } else {
    energyLastTime = energy;
    return true;
  }
}

void MotionSegmentation::generateMotionProposals(
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>> &motionProposals) {
  //为outlier标签的���给一个空的fundamental
  // matrix，将其放�������������������一列������������������������������������
  //就是为了占一���位置而已。
  motionProposals.push_back(Eigen::MatrixXf::Zero(3, 3));
  // 首先应���将outlier全�����������������去���，然后用剩下的点来提取proposal。
  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {

      // if (outliers_(y * width_ + x, 0) == 0) {
      points1.push_back(cv::Point2f(baseVec_(x + y * width_, 0),
                                    baseVec_(x + y * width_, 1)));

      points2.push_back(cv::Point2f(matchVec_(x + y * width_, 0),
                                    matchVec_(x + y * width_, 1)));

      // std::cout << "(" << baseVec_(x + y * width_, 0) << ","
      //           << baseVec_(x + y * width_, 1) << std::endl;
      // std::cout << "(" << matchVec_(x + y * width_, 0) << ","
      //           << matchVec_(x + y * width_, 1) << std::endl;
      // std::cout << "-------" << std::endl;
      // cv::waitKey(3000);
      // }
    }
  }

  std::vector<uchar> inliers; //(points1.size(), 0);
  int oldInlierNumber;
  while (points1.size() > initThreshold && oldInlierNumber != points1.size()) {
    oldInlierNumber = points1.size();
    cv::Mat fundamental =
        cv::findFundamentalMat(points1, points2, CV_FM_LMEDS, 3, 0.99, inliers);

    // cv::Mat R;
    // cv::Mat K(3, 3, CV_64F);
    // cv::Mat W(3, 3, CV_64F);
    // K = (cv::Mat_<double>(3, 3) << 721.537700, 0.000000, 609.559300,
    // 0.000000,
    //      721.537700, 172.854000, 0.000000, 0.000000, 1.000000);
    // cv::SVD svd(K.t() * fundamental * K, cv::SVD::MODIFY_A);
    // W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    // R = svd.u * W * svd.vt;
    // std::cout << "R:" << std::endl << R << std::endl;
    fundamental.convertTo(fundamental, CV_32F);
    Eigen::Matrix3f fundamentalMatrix;
    cv2eigen(fundamental, fundamentalMatrix);

    if (fundamentalMatrix.array().abs().sum() <= 0.0) {
      std::cout << "there is a bug appear when generating motion proposal..."
                << std::endl;
      break;
    }
    std::cout << "the remaining inlier points:" << points1.size() << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout.precision(8);
    std::cout << fundamentalMatrix << std::endl;
    motionProposals.push_back(fundamentalMatrix);
    std::vector<cv::Point2f> temp1, temp2;
    std::vector<uchar>::const_iterator itIn = inliers.begin();
    std::vector<cv::Point2f>::const_iterator it1 = points1.begin();
    std::vector<cv::Point2f>::const_iterator it2 = points2.begin();
    // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    // indi(
    //     436, 1024);
    // indi.setZero();

    for (; itIn != inliers.end(); itIn++, ++it1, ++it2) {
      if (*itIn == false) {
        temp1.push_back(*it1);
        // std::cout << it1->y << "," << it1->x << std::endl;
        // indi(it1->y, it1->x) = 1;
        temp2.push_back(*it2);
      }
    }
    // cv::Mat result;
    // cv::eigen2cv(indi, result);
    // cv::imshow("indicator_outliers for motion estimation", result);
    // cv::waitKey(0);
    points1.swap(temp1);
    points2.swap(temp2);
    inliers.clear();
    temp1.clear();
    temp2.clear();
  }
}

void MotionSegmentation::recoverHardAssignment(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u) {
  // recover hard assignment from u；
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      segmentationResult(u.rows(), 1);
  // int *motionLabel = segmetationResult.ptr<int>(0);
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < u.rows(); ++i) {
    Eigen::MatrixXf::Index maxU[1];
    u.row(i).maxCoeff(&maxU[0]);
    u.row(i).setZero();
    u(i, maxU[0]) = 1;
    segmentationResult(i, 0) = maxU[0];
  }

  //转换为hard assignment之后顺便再加一个显示功能。
  //直接将列号当做label输出了。
  segmentationResult.resize(height_, width_);
  // std::cout << segmentationResult.block(0, 0, 20, 20) << std::endl;
  // cv::waitKey(0);
  cv::Mat result;
  cv::eigen2cv(segmentationResult, result);
  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> conv;
  // conv = u.col(1);
  // conv.resize(height_, width_);
  // std::cout << conv.block(0, 0, 20, 20) << std::endl;
  // cv::eigen2cv(conv, result);

  // result.convertTo(result, CV_32SC1);
  // writeData("result.txt", result);
  // std::cout << "data has been writen..." << std::endl;
  // cv::waitKey(0);
  cv::Mat colorLabelImage;
  labelColor(result, colorLabelImage);
  cv::imshow("Motion segmentation result...", colorLabelImage);
  cv::waitKey(1);
}

void MotionSegmentation::modelDiscovery(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u,
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>> &motionProposals) {
  cv::Mat outlierLabel;
  outlierLabel = ccaByTwoPass(u.col(0));
  double minVal = 0, maxVal = 0;
  cv::minMaxIdx(outlierLabel, &minVal, &maxVal);
  std::vector<std::vector<cv::Point2f>> outlierClusters;
  outlierClusters.resize((int)maxVal);
  for (int i = 0; i < outlierLabel.rows; ++i) {
    int *label = outlierLabel.ptr<int>(i);
    for (int j = 0; j < outlierLabel.cols; ++j) {
      if (label[j] != 0) {
        cv::Point2f point;
        point.x = j;
        point.y = i;
        outlierClusters[label[j] - 1].push_back(point);
      }
    }
  }

  //根据outlier 连通域结果，挖掘新的Motion proposal
  for (std::vector<std::vector<cv::Point2f>>::iterator it =
           outlierClusters.begin();
       it < outlierClusters.end(); ++it) {
    if ((*it).size() > outlierDiscoveryTh) { //以后设置成8
      std::vector<cv::Point2f> point1;
      std::vector<cv::Point2f> point2;
      for (std::vector<cv::Point2f>::iterator itP = (*it).begin();
           itP < (*it).end(); ++itP) {
        point1.push_back(*itP);
        cv::Point2f pt;
        pt.x = matchVec_((*itP).x + (*itP).y * width_, 0);
        pt.y = matchVec_((*itP).x + (*itP).y * width_, 1);
        point2.push_back(pt);
      }
      std::vector<uchar> mask;
      cv::Mat fundamental =
          cv::findFundamentalMat(point1, point2, CV_FM_LMEDS, 3, 0.99, mask);
      std::cout << "using " << mask.size()
                << " points to generate motion proposal..." << std::endl;
      fundamental.convertTo(fundamental, CV_32F);
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      fundamentalMatrix(fundamental.ptr<float>(), fundamental.rows,
                        fundamental.cols);
      Eigen::MatrixXf indicator;
      indicator.setZero(3, 3);
      indicator = (fundamentalMatrix.array() != 0).select(indicator, 1);
      if (indicator.sum() < 9) {
        //������������一�������Motion
        // proposal不�����������������������������空���压�����
        motionProposals.push_back(fundamentalMatrix);
        // std::cout << fundamentalMatrix << std::endl;
      } else {
        std::cout << "too few points to generate new outlier proposal..."
                  << std::endl;
      }
    }
  }
  std::cout << "start inlier motion model discovery..." << std::endl;
  std::cout << "===================================================="
            << std::endl;
  for (int inlierIdx = 1; inlierIdx < u.cols(); ++inlierIdx) {
    cv::Mat inlierLabel = ccaByTwoPass(u.col(inlierIdx));
    minVal = 0, maxVal = 0;
    cv::minMaxIdx(inlierLabel, &minVal, &maxVal);
    //首先�������������������下这个标签是不是已经只有一坨点在用，
    if (maxVal > 1) {
      std::vector<std::vector<cv::Point2f>> inlierClusters;
      inlierClusters.resize((int)maxVal);
      for (int i = 0; i < inlierLabel.rows; ++i) {
        int *label = inlierLabel.ptr<int>(i);
        for (int j = 0; j < inlierLabel.cols; ++j) {
          if (label[j] != 0) {
            cv::Point2f point;
            point.x = j;
            point.y = i;
            inlierClusters[label[j] - 1].push_back(point);
          }
        }
      }

      for (std::vector<std::vector<cv::Point2f>>::iterator it =
               inlierClusters.begin();
           it < inlierClusters.end(); ++it) {
        if ((*it).size() > outlierDiscoveryTh) {
          std::vector<cv::Point2f> point1;
          std::vector<cv::Point2f> point2;
          for (std::vector<cv::Point2f>::iterator itP = (*it).begin();
               itP < (*it).end(); ++itP) {
            point1.push_back(*itP);
            cv::Point2f pt;
            pt.x = matchVec_((*itP).x + (*itP).y * width_, 0);
            pt.y = matchVec_((*itP).x + (*itP).y * width_, 1);
            point2.push_back(pt);
          }
          std::vector<uchar> mask;
          cv::Mat fundamental = cv::findFundamentalMat(
              point1, point2, CV_FM_LMEDS, 3, 0.99, mask);
          std::cout << "using " << mask.size()
                    << " points to generate motion proposal..." << std::endl;
          fundamental.convertTo(fundamental, CV_32F);
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
          fundamentalMatrix(fundamental.ptr<float>(), fundamental.rows,
                            fundamental.cols);
          Eigen::MatrixXf indicator;
          indicator.setZero(3, 3);
          indicator = (fundamentalMatrix.array() != 0).select(indicator, 1);
          if (indicator.sum() < 9) {
            //判����������下Motion proposal不为空再压栈
            motionProposals.push_back(fundamentalMatrix);
            // std::cout << fundamentalMatrix << std::endl;
          } else {
            std::cout << "inlier points for LMEDS cannot generate correct "
                         "proposal..."
                      << std::endl;
          }
        }
      }
    }
  }
  // update u
  u.conservativeResize(Eigen::NoChange, motionProposals.size());
  u.setConstant(1.0 / u.cols());
  // for (int i = 0; i < u.rows(); ++i) {
  //   if (outliers_(i, 0) == 1) {
  //     u.row(i).setZero();
  //     u(i, 0) = 1.0;
  //   }
  // } // update finish.改
}

cv::Mat MotionSegmentation::ccaByTwoPass(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        outlier) {
  // connected componet analysis(4-component)
  // background = 0,forground = 1;
  //这里为了安全也可以加一个报错���数。
  outlier.resize(height_, width_);
  // first pass
  cv::Mat outliers;
  cv::eigen2cv(outlier, outliers);
  outliers.convertTo(outliers, CV_32SC1);
  int label = 1; //背景标签是0，前景的标签从1开始计算。
  std::vector<int> labelSet;
  labelSet.push_back(0);
  labelSet.push_back(1);

  for (int i = 1; i < outliers.rows - 1; ++i) {
    int *dataPreRow = outliers.ptr<int>(i - 1);
    int *dataCurRow = outliers.ptr<int>(i);
    for (int j = 1; j < outliers.cols - 1; ++j) {
      if (dataCurRow[j] == 1) {
        std::vector<int> neighborLabels;
        neighborLabels.reserve(2);
        int leftPixel = dataCurRow[j - 1];
        int upPixel = dataPreRow[j];
        if (leftPixel > 1) {
          neighborLabels.push_back(leftPixel);
        }
        if (upPixel > 1) {
          neighborLabels.push_back(upPixel);
        }
        if (neighborLabels.empty()) {
          labelSet.push_back(++label);
          dataCurRow[j] = label;
          labelSet[label] = label;
        } else {
          std::sort(neighborLabels.begin(), neighborLabels.end());
          int smallestLabel = neighborLabels[0];
          dataCurRow[j] = smallestLabel;
          // save equivalence
          for (size_t k = 1; k < neighborLabels.size(); ++k) {
            int tempLabel = neighborLabels[k];
            int &oldSmallestLabel = labelSet[tempLabel];
            if (oldSmallestLabel > smallestLabel) {
              labelSet[oldSmallestLabel] = smallestLabel;

            } else if (oldSmallestLabel < smallestLabel) {
              labelSet[smallestLabel] = oldSmallestLabel;
            }
          }
        }
      }
    }
  }
  // update equivalent labels
  // assigned with the smallest label in each equivalent label set
  for (size_t i = 2; i < labelSet.size(); i++) {
    int curLabel = labelSet[i];
    int preLabel = labelSet[curLabel];
    while (preLabel != curLabel) {
      curLabel = preLabel;
      preLabel = labelSet[preLabel];
    }
    labelSet[i] = curLabel;
  }
  // 2. second pass
  for (int i = 0; i < outliers.rows; i++) {
    int *data = outliers.ptr<int>(i);
    for (int j = 0; j < outliers.cols; j++) {
      int &pixelLabel = data[j];
      pixelLabel = labelSet[pixelLabel];
    }
  }

  // writeData("outlier.txt", outliers);

  // cv::Mat colorLabelImage;
  // labelColor(outliers, colorLabelImage);
  // cv::imshow("connected componet analysis(4-component)", colorLabelImage);
  // cv::waitKey(0);
  return outliers;
}

// void MotionSegmentation::modelClustering(
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &u,
//     std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
//                               Eigen::RowMajor>> &motionProposals) {
//   //这个位置要设计一个东西，让Motion model能够相似的就合并。
//
//   std::vector<
//       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
//       newProposals;
//   newProposals.push_back(Eigen::MatrixXf::Zero(3, 3));
//   //可以采用比较出栈的方式来进行比较，就是先让第一个和其余的比，然后将比base更合适的放在第一位，
//   for (int i = 1; i < motionProposals.size(); ++i) {
//     Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//     fBase;
//     fBase = motionProposals[i];
//     fBase.resize(9, 1);
//
//     for (int j = 1; j < motionProposals.size(); ++j) {
//       Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//           fMatch, top, bot, similarity;
//       fMatch = motionProposals[j];
//       fMatch.resize(9, 1);
//
//       top = fBase * fMatch;
//       bot = fBase.square().colwise().sum().sqrt() *
//             fMatch.square().colwise().sum().sqrt();
//       similarity = top.sum() / bot;
//       std::cout << "the similarity between num:" << i << "&num:" << j
//                 << "proposal is:" << similarity(0) << std::endl;
//       if (0.99 < similarity(0) < 1) {
//         if (u.col(i).sum() < u.col(j).sum()) {
//           newProposals.push_back(motionProposals[j]);
//
//           break;
//         }
//       }
//     }
//   }
//   cv::waitKey(0);
//   // u.conservativeResize(Eigen::NoChange, motionProposals.size());
//   // u.setConstant(1.0 / u.cols());
//   // for (int i = 0; i < u.rows(); ++i) {
//   //   if (outliers_(i, 0) == 1) {
//   //     u.row(i).setZero();
//   //     u(i, 0) = 1.0;
//   //   }
//   // } // update finish.
// }