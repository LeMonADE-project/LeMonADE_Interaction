# build and run LeMonaDE-ROP-tests
INCLUDE_DIRECTORIES("${source_dir}/include")
# FILE (GLOB_RECURSE test_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)
FILE (GLOB_RECURSE test_SRCS *.cpp )
SET (test_LIBS LeMonADE ) # add more libraries if needed 
SET (test_BIN ${PROJECT_NAME}-tests)

# configure_file(BondCreationBreaking.dat   BondCreationBreaking.dat  COPYONLY) # to copy some bfm files to the test directory
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/tests)


IF (NOT CMAKE_CROSSCOMPILING)
    LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
    ADD_EXECUTABLE(${test_BIN} ${test_SRCS})
    TARGET_LINK_LIBRARIES(${test_BIN} ${test_LIBS})
    MESSAGE("Running ${PROJECT_NAME} tests...")
    MESSAGE("Test sources are  ${test_SRCS}" )
    ADD_CUSTOM_TARGET(check ALL "${CMAKE_BINARY_DIR}/tests/${test_BIN}" DEPENDS ${test_BIN} COMMENT "Executing ${PROJECT_NAME} tests..." VERBATIM SOURCES ${test_SRCS})
    ADD_CUSTOM_TARGET(testrun "${CMAKE_BINARY_DIR}/tests/${test_BIN}" DEPENDS ${test_BIN} COMMENT "Executing ${PROJECT_NAME} tests..." VERBATIM SOURCES ${test_SRCS})
ENDIF (NOT CMAKE_CROSSCOMPILING)



