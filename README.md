# DNN Inference Engine
Projects for CS492 Systems for Machine Learning, Spring 2020, KAIST

## About
This repository is for projects which are implementations of inference engine for DNN that consists of layers such as convolution, batch normalization, and max pooling.

## Projects

### proj1
Runs inference of `yolov2tiny` model using existing library (`tensorflow` API).

### proj2
Implementation of DNN inference engine using only python `numpy` API.

### proj3
Implementation of DNN inference engine using various parallelization techniques. Runs inference of `yolov2tiny` with the engines using python `ctypes`.
- CPU parallelization
    - High level library: `OpenBLAS`
    - Manual: `AVX`, `pthread`
- GPU parallelization
    - High level library: `cuBLAS`
    - Manual: `CUDA`
