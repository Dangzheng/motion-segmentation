//首先是各种阈值
// initial points threshold at generate motion proposal;
int initThreshold = 100;

// Wl beta;越大的话，图像灰度边缘处越能容忍大的变化；
// 越小越不能容忍变化。
float beta = 10; // 0.004;
// outlier constant cost
// gamma
// forward backward consistant check threshold;
float checkThreshold = 3.0;

// outlier model discovery threshold;
int outlierDiscoveryTh = 1000;
// inlier model discovery threshold;
int inlierDiscoveryTh = 1000;