## LeMonADE - Project Name 

This version is build on the develop fork of [LeMonADE](https://github.com/tonimueller).

### Authors 
* Toni Mueller

### Installation ###
1. Install the LeMonADE directory.
2. Execute the 'configure' script with the following options:
 *  -DLEMONADE_DIR=/path/to/LeMonADE-Installation/
 *  -DBUILDDIR=/build/directory/  Default is ./build
 *  -DLEMONADE_TESTS=ON/OFF Default is :OFF
 *  -DCMAKE_BUILD_TYPE=Release/Debug Default is : Release
 For the most cases the default is enough and one needs to execute:
```shell
 ./configure -DLEMONADE_DIR=/path/to/LeMonADE-Installation/
```

### Idea or project discription 
This template repository can be downloaded as a first step to start a new project. 
The simulation code should be placed in the projects directory (of course with some
subfolders) and it is highly recommended to write tests. 

### How to use it?
* The _CommandLineParser_ is a simple way to hand over parameters to the program. There
are several alternative like the Boost library ( see 
[tutorial](https://theboostcpplibraries.com/boost.program_options)) or the native 
[iostream library](https://www.cplusplus.com/articles/DEN36Up4/) in cpp.
* Here the [catch2](https://github.com/catchorg/Catch2) testsuit is used. See the 
homepage for further information how to use it.


### License
See [LeMonADE-license](https://github.com/LeMonADE-project/LeMonADE/blob/master/LICENSE)

