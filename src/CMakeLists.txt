# ADD_SUBDIRECTORY(LeMonADE_Interaction)
# ---------------------------------------------------------------------------------
#     ooo      L   attice-based  |
#   o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
#  o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
# oo---0---oo  A   lgorithm and  |
#  o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
#   o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
#     ooo                        |
# ---------------------------------------------------------------------------------
#
# This file is part of LeMonADE.
#
# LeMonADE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LeMonADE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------------
ADD_SUBDIRECTORY(LeMonADE_Interaction)
SET ( project_SRCS
	${CMAKE_CURRENT_SOURCE_DIR}/LeMonADE_Interaction/updater/UpdaterGPU_Interaction.cu
	# ${CMAKE_CURRENT_SOURCE_DIR}/utility/CommandlineParser.cpp
)  
     
MESSAGE( "Add src files : " ${project_SRCS})
SET (project_BIN ${PROJECT_NAME})

IF (NOT DEFINED project_build_static)
   SET(project_build_static 1)
ENDIF (NOT DEFINED project_build_static)

IF (project_build_static)
	CUDA_ADD_LIBRARY(libLeMonADE_Interaction-static STATIC ${project_SRCS} )
	SET_TARGET_PROPERTIES(libLeMonADE_Interaction-static PROPERTIES CUDA_SEPARABLE_COMPILATION ON VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME ${project_BIN} CLEAN_DIRECT_OUTPUT 1)
	INSTALL(TARGETS libLeMonADE_Interaction-static DESTINATION lib)
ENDIF(project_build_static)

