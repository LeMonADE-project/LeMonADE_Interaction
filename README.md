## LeMonADE - Radicalic Dimer Polymerisation 

This version is build on the develop fork of [LeMonADE](https://github.com/tonimueller).

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
# Parameter
* activation energy 
* concentration of radicalic dimers 

# Analyzes 
* molecular weight distribution or equivalent raw data to produce this ( a bond table ) 

# Known problems 
* dimers which are connected could cause to curruptions in the bfm file, because consecutive 
monomers would written into one line in the mcs write command. BUT connections across periodic 
boundaries create bond greater square length 10...

