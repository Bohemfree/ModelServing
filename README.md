# C++ ModelServing
Convert a Keras model(Unet) created in Python to be used in C++  
Tensor conversion using [cppflow github](https://github.com/serizba/cppflow)

------
Build tensorflow steps  
1. Install [Bazel](https://docs.bazel.build/versions/main/install.html) needed to compile tensorflow 
2. Install [MSYS2](https://www.msys2.org/) needed to build tensorflow 
3. Install and Build specific [tensorflow](https://github.com/tensorflow/tensorflow) version to your spec
4. Build dll and lib files:
* bazel build --config=cuda tensorflow:tensorflow.dll
* bazel build --config=cuda tensorflow:tensorflow.lib
----
Run steps
1. Install [cppflow library](https://github.com/serizba/cppflow)
2. Register opencv, tensorflow ,cppflow library and tensorflow.dll, tensorflow.lib
3. Convert keras model(hdf5 format) to pb format model for using in C++ (hdf52pb.py)
4. Run cppflow_test.cpp


Environment  
- Windows 10
- tensorflow : 2.3.4
- python : 3.6.8
- cuda : 10.1
- cudnn : 7.6.5

Reference  
- https://www.tensorflow.org/install/source_windows
