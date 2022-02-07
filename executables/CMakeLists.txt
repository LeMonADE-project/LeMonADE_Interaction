ADD_EXECUTABLE(generate_blockcopolymers generate_blockcopolymers.cpp)
TARGET_LINK_LIBRARIES(generate_blockcopolymers LeMonADE CommandlineParser)
# cpu based interaction simulator 
ADD_EXECUTABLE(simulate_interaction simulate_interaction.cpp)
TARGET_LINK_LIBRARIES(simulate_interaction LeMonADE CommandlineParser)


# cuda_add_executable( simulate_interactionGPU simulate_interactionGPU.cpp )

# if( ( NOT DEFINED project_build_static ) OR ( project_build_static ) )
#     target_link_libraries(simulate_interactionGPU libLeMonADE_Interaction-static LeMonADEGPU LeMonADE cuda CommandlineParser )
# else()
#     link_directories ("${CMAKE_BINARY_DIR}/lib/")
#     target_link_libraries(simulate_interactionGPU LeMonADEGPU libLeMonADE_Interaction LeMonADE cuda CommandlineParser )
# endif()