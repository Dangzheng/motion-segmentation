# Motion segmentation part

## descriptions:

These codes were written according to the paper.

```
Dense Monocular Depth Estimation in Complex Dynamic Scenes
Rene Ranftl, Vibhav Vineet, Qifeng Chen, Vladlen Koltun 
```
I only implement the **motion segmentation part**, code was tested on sintel dataset flow. **Reconstruction part** can be found at [github](https://github.com/cvfish/VideoPopup).

Some functions have a variety of options on the algorithm.

The code **not use GPU**. Primal-dual use entopic setting instead of euclidean setting according to the parper.

```
On the ergodic convergence rates of a first-order primal-dual algorithm
Antonin Chambolle, Thomas Pock
```

## environment and dependencies

* opencv 2.4.13
* eigen
* openMP
* language: C/C++, matlab(only some tools)
* tested MAC 10.12

images and calibration results were get form Sintel flow (temple2).

## details

* change CMakeLists.txt first(if you don't like openMP, disable it and delete relative part in the code, most of them is begin with #program.)
* make
* ./test frame_0020.png

### about the input
* Both forward and backward optical results are required. 

```
Flow Fields: Dense Correspondence Fields for Highly Accurate Large Displacement Optical Flow Estimation
Christian Bailer, Bertram Taetz, Didier Stricker
```
```
EpicFlow: Edge-Preserving Interpolation of Correspondences for Optical Flow
Jerome Revauda, Philippe Weinzaepfela, Zaid Harchaouia, Cordelia Schmida
```
Code from both of the paper have been used to get optical result. Full size image will cost a lot of time, so I provide matlab code to downsize the optical results and input image. Full size data has been contained in full data folder.

* optical flow result example(foreward and backward)

![fore](https://github.com/Dangzheng/motion-segmentation/raw/master/full_data/fore.png)
 
![back](https://github.com/Dangzheng/motion-segmentation/raw/master/full_data/back.png)

* input image(left 1/4 size)

![left_img](https://github.com/Dangzheng/motion-segmentation/raw/master/full_data/frame_0020.png)

* output image(1/4 size)

![out](https://github.com/Dangzheng/motion-segmentation/raw/master/full_data/out.png)