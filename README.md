## LeMonADE - Interaction

This repo is concerned with the simulation of systems with a pairwise interaction.

### Authors 
Seed [Authors](https://github.com/LeMonADE-project/LeMonADE_Interaction/blob/master/AUTHORS.md)

### Installation ###
1. Install LeMonADE from the develop branch of https://github.com/LeMonADE-project/LeMonADE.git.
2. Install LEMonADE-GPU from the master branch of https://github.com/LeMonADE-project/LeMonADE-GPU.git.
3. Execute the 'configure' script with the following options:
 *  -DLEMONADE_DIR=/path/to/LeMonADE-Installation/
 *  -DLEMONADEGPU_DIR=/path/to/LeMonADEGPU-Installation/
 *  -DCUDA_ARCH=Compute_Capability_For_GPU
 *  -DBUILDDIR=/build/directory/  Default is ./build
 *  -DLEMONADE_TESTS=ON/OFF Default is :OFF
 *  -DCMAKE_BUILD_TYPE=Release/Debug Default is : Release
 For the most cases the default is enough and one needs to execute:
```shell
 ./configure -DLEMONADE_DIR=/path/to/LeMonADE-Installation/
```
4. Compile the project by executing 
```shell
  make 
```
Compute capability can be found in : https://developer.nvidia.com/cuda-gpus

### How to use it?
* The _CommandLineParser_ is a simple way to hand over parameters to the program. There
are several alternative like the Boost library ( see 
[tutorial](https://theboostcpplibraries.com/boost.program_options)) or the native 
[iostream library](https://www.cplusplus.com/articles/DEN36Up4/) in cpp.
* Here the [catch2](https://github.com/catchorg/Catch2) testsuit is used. See the 
homepage for further information how to use it.


### License
See [LeMonADE-license](https://github.com/LeMonADE-project/LeMonADE/blob/master/LICENSE)

