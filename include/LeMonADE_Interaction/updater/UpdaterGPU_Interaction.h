/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo--GPU--oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADEGPU.

LeMonADE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LeMonADE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------*/
#ifndef LEMONADEINTERACTION_UPDATER_UPDATERGPUINTERACTION_H
#define LEMONADEINTERACTION_UPDATER_UPDATERGPUINTERACTION_H

#include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>


template< typename T_UCoordinateCuda >
class UpdaterGPU_Interaction: public UpdaterGPUScBFM<T_UCoordinateCuda> {
public:
    typedef UpdaterGPUScBFM< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM<  uint8_t > :: T_Flags      ;
    using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;

    typedef uint32_t T_InteractionLattice;
    typedef uint8_t T_InteractionTag;

    using BaseClass::mLog;
    
protected:
    using BaseClass::mBoxX;
    using BaseClass::mBoxY;
    using BaseClass::mBoxZ;
    using BaseClass::mBoxXM1;
    using BaseClass::mBoxYM1;
    using BaseClass::mBoxZM1;
    using BaseClass::mBoxXLog2;
    using BaseClass::mBoxXYLog2;
    using BaseClass::met;
    using BaseClass::mStream;
    using BaseClass::mPolymerSystem;
    using BaseClass::mnAllMonomers;
    using BaseClass::mNeighbors;
    using BaseClass::checkBondVector;
    using BaseClass::mviPolymerSystemSortedVirtualBox;
    using BaseClass::mPolymerSystemSortedOld;
    using BaseClass::mPolymerSystemSorted;
    using BaseClass::mnElementsInGroup;
    using BaseClass::mCudaProps;
    using BaseClass::mAge;
    using BaseClass::mUsePeriodicMonomerSorting;
    using BaseClass::mnStepsBetweenSortings;
    using BaseClass::doSpatialSorting;
    using BaseClass::useOverflowChecks;
    using BaseClass::findAndRemoveOverflows;
    using BaseClass::mnLatticeTmpBuffers;
    using BaseClass::mLatticeTmp;
    using BaseClass::mvtLatticeTmp;
    using BaseClass::randomNumbers;
    using BaseClass::launch_PerformSpeciesAndApply;
    using BaseClass::launch_PerformSpecies;
    using BaseClass::launch_ZeroArraySpecies;
    using BaseClass::miNewToi;
    using BaseClass::miToiNew;
    using BaseClass::mviSubGroupOffsets;
    using BaseClass::mNeighborsSorted;
    using BaseClass::mNeighborsSortedInfo;
    using BaseClass::mGroupIds;
    using BaseClass::mNeighborsSortedSizes;
    using BaseClass::hGlobalIterator;
    using BaseClass::doCopyBackMonomerPositions;
    using BaseClass::doCopyBackConnectivity;
    using BaseClass::diagMovesOn;
    using BaseClass::boxCheck;
    using BaseClass::checkSystem;

    //! Interaction energies between monomer types. Max. type=255 given by max(uint8_t)=255
    double interactionTable[256][256];

    //! Lookup table for exp(-interactionTable[a][b])
    double probabilityLookup[256][256];

    
public:
    UpdaterGPU_Interaction();
    ~UpdaterGPU_Interaction();
private:
    //create a lattice with the ids on the edges
    MirroredTexture< T_InteractionTag > * mLatticeInteractionTag;
    
    // std::vector< T_Id >  mNewToOldReactiveID;

    //store the inteaction tag (needed in the init step)
    //abuse the already existing attribute vector for this purpose 
    // std::vector<T_InteractionTag> interactionTag;
    //attributeTag is available only on CPU side
    //but for the lattice setup we need a GPU available vector
    //TODO: work this out in the init function 
    MirroredVector< T_InteractionTag > * mInteractionTag;
    

public:
    void initialize();
    void runSimulationOnGPU(const uint32_t nSteps );
    void checkInteractionLatticeOccupation() ; 
    void cleanup();
    void destruct();

    /** set the interaction tag in std::vector interactionTag
    * to keep it simple we use instead the already existing 
    * attribute container of the main simulator ;-)
    **/ 
    void setInteractionTag(uint32_t id, T_InteractionTag tag );
    //set the nearest neighbor interaction for the gpu 
    void setNNInteraction(int32_t typeA, int32_t typeB, double energy);
    //get the nearest neighbor interaction for the gpu 
    double getNNInteraction(int32_t typeA, int32_t typeB) const; 

    void launch_CheckConnection(
	  const size_t nBlocks, const size_t nThreads, 
      const size_t iSpecies, const uint64_t seed);
    void launch_ApplyConnection(
	  const size_t nBlocks , const size_t   nThreads, 
	  const size_t iSpecies);


    void initializeInteractionLattice();
    void launch_initializeInteractionLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );
    void launch_resetInteractionLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );

};
#endif
