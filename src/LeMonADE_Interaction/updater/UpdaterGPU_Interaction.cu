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
/*
 * UpdaterGPU_Interaction.cu
 *
 *  Created on: 04.02.2022
 *     Authors: Toni Mueller
 */
#include <LeMonADE_Interaction/updater/UpdaterGPU_Interaction.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <cuda_profiler_api.h>              // cudaProfilerStop
#include <LeMonADEGPU/utility/AutomaticThreadChooser.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/graphColoring.tpp>
#include <LeMonADEGPU/core/rngs/Saru.h>
#include <LeMonADEGPU/core/MonomerEdges.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/feature/BoxCheck.h>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <LeMonADEGPU/core/BondVectorSet.h>
using T_Flags            = UpdaterGPU_Interaction< uint8_t >::T_Flags         ;
using T_Id               = UpdaterGPU_Interaction< uint8_t >::T_Id            ;
using T_InteractionTag   = UpdaterGPU_Interaction< uint8_t >::T_InteractionTag;
__device__ __constant__ uint32_t DXTableNN_d[18];
__device__ __constant__ uint32_t DYTableNN_d[18];
__device__ __constant__ uint32_t DZTableNN_d[18];
__device__ __constant__ double dcNNProbability[256][256];
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////Defintion of member functions for the interaction lattice //////////////
///////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::initializeInteractionLattice(){
    if ( mLatticeInteractionTag != NULL ){
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initializeInteractionLattice] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    size_t nBytesLatticeTmp = mBoxX * mBoxY * mBoxZ * sizeof(T_InteractionTag);
     mLog( "Info" ) << "Allocate "<< nBytesLatticeTmp/1024<<"kB  memory for lattice \n";  
    mLatticeInteractionTag  = new MirroredTexture< T_InteractionTag >( nBytesLatticeTmp, mStream );
}

/**
 * @brief writes the ID of the chain ends on the lattice
 * @details The ID start at 1 and are shifted by and offset which is given
 * 	    by the previous species of monomers. 
 */
template< typename T_UCoordinateCuda >
__global__ void kernelUpdateInteractionLattice
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        const * const __restrict__ dpPolymerSystem     ,
    T_InteractionTag    const * const __restrict__ dInteractionTag     ,                
    uint32_t            const                      iOffset             ,
    T_InteractionTag          * const __restrict__ dpInteractionLattice,
    T_Id                        const              nMonomers           ,
    Method                      const              met 
){
    for ( T_Id id = blockIdx.x * blockDim.x + threadIdx.x;
          id < nMonomers; id += gridDim.x * blockDim.x ){
        auto const r0 = dpPolymerSystem[ iOffset + id ];
        auto const interactionTag= dInteractionTag[iOffset + id];
		dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y  , r0.z   ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y+1, r0.z   ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y  , r0.z+1 ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y+1, r0.z+1 ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y  , r0.z   ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y+1, r0.z   ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y  , r0.z+1 ) ] = ( interactionTag+1 );
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y+1, r0.z+1 ) ] = ( interactionTag+1 );
    }
}
 /**
  * @brief convinience function to update the lattice occupation. 
  * @details We introduce such functions because then they can be used latter on from inheritate classes..
  */
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::launch_initializeInteractionLattice(
  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies ){
	mLog ( "Check" ) <<"Start filling lattice with ones:  \n" ;
	if ( false ){ //fill in cpu 
		mPolymerSystemSorted->pop();
		for (T_Id i =0; i < mnElementsInGroup[ iSpecies ]; i++){
			auto const id(i+mviSubGroupOffsets[ iSpecies ]);
			auto const r(mPolymerSystemSorted->host[id]); 
			auto const Vector(met.getCurve().linearizeBoxVectorIndex(r.x,r.y,r.z));
			mLatticeInteractionTag->host[Vector]= mInteractionTag->host[id]+1;
		}
		mLatticeInteractionTag->push(0);
		cudaStreamSynchronize( mStream );
	}else{
		kernelUpdateInteractionLattice<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
			mPolymerSystemSorted->gpu     ,         
            mInteractionTag->gpu          ,   
			mviSubGroupOffsets[ iSpecies ], 
			mLatticeInteractionTag->gpu   ,
			mnElementsInGroup[ iSpecies ] ,                        
			met
		);
	}
}
/**
 * @brief writes 0 on the lattice where the chain ends are located 
 * @details Using this brings performance, because the lattice is dilute
 */
template< typename T_UCoordinateCuda >
__global__ void kernelResetInteractionLattice
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        const * const __restrict__ dpPolymerSystem  ,
    uint32_t            const                      iOffset          ,
    T_InteractionTag          * const __restrict__ dpInteractionLattice,
    T_Id                        const              nMonomers        ,
    Method                      const              met 
){
    for ( T_Id idB = blockIdx.x * blockDim.x + threadIdx.x;
          idB < nMonomers; idB += gridDim.x * blockDim.x ){
        auto const r0 = dpPolymerSystem[ iOffset + idB ];
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y  , r0.z   ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y+1, r0.z   ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y  , r0.z+1 ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x  , r0.y+1, r0.z+1 ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y  , r0.z   ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y+1, r0.z   ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y  , r0.z+1 ) ] = T_InteractionTag(0);
        dpInteractionLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x+1, r0.y+1, r0.z+1 ) ] = T_InteractionTag(0);
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::launch_resetInteractionLattice(
  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies ){
	if ( false ){ //erasse in cpu 
		mPolymerSystemSorted->pop();
		for (T_Id i =0; i < mnElementsInGroup[ iSpecies ]; i++){
			auto const idB(i+mviSubGroupOffsets[ iSpecies ]);
			auto const r(mPolymerSystemSorted->host[idB]); 
			auto const Vector(met.getCurve().linearizeBoxVectorIndex(r.x,r.y,r.z));
			mLatticeInteractionTag->host[Vector]= 0;
		}
		mLatticeInteractionTag->push(0);
		cudaStreamSynchronize( mStream );
	}else{
		kernelResetInteractionLattice<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
		mPolymerSystemSorted->gpu     ,            
		mviSubGroupOffsets[ iSpecies ], 
		mLatticeInteractionTag->gpu              ,
		mnElementsInGroup[ iSpecies ] ,                        
		met
		);
	}
}
/**
 * @brief Counts the number of occupied lattice sites.
 */
template< typename T_UCoordinateCuda  >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::checkInteractionLatticeOccupation()  
{
	mLatticeInteractionTag->pop(0);
	uint32_t countLatticeEntries(0);
	for(T_Id x=0; x< mBoxX; x++ )
		for(T_Id y=0; y< mBoxY; y++ )
			for(T_Id z=0; z< mBoxX; z++ )
				if(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)] > 0 )
					countLatticeEntries++;
	assert(mnAllMonomers*8 == countLatticeEntries );  
	mLog( "Check" )
		<< "checkInteractionLatticeOccupation: \n"
		<< "mnAllMonomers       = " <<       mnAllMonomers << "\n"
		<< "countLatticeEntries = " << countLatticeEntries << "\n";
    //TODO:teseting the consistency of the lattice and the specied position
	// mPolymerSystemSorted->pop();
	// for(T_Id x=0; x< mBoxX; x++ )
	// 	for(T_Id y=0; y< mBoxY; y++ )
	// 		for(T_Id z=0; z< mBoxX; z++ ){
	// 			T_InteractionTag LatticeEntry(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)]);
	// 			if( LatticeEntry > 0 ){
	// 				auto r=mPolymerSystemSorted->host[LatticeEntry-1 + mviSubGroupOffsets[1] ];
	// 				if ( r.x %512  != x || r.y %512 != y || r.z %512!= z  ){
	// 					std::stringstream error_message;
	// 					error_message << "LatticeEntry=  "<<LatticeEntry  << " "
	// 						<< "Pos= ("<< x <<"," << y << "," << z << ")" << " "
	// 						<< "mPolymerSystemSorted= ("<< r.x <<"," << r.y << "," << r.z << ")" << "\n";
	// 					throw std::runtime_error(error_message.str());
	// 				}
	// 			}
	// 		}
}
///////////////////////////////////////////////////////////////////////////////
//Lattice handling is done ////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////
__device__ inline double getProbability(uint32_t typeA, uint32_t typeB){
    return dcNNProbability[typeA][typeB];
}
__device__ inline double calcInteractionProbability(
    T_InteractionTag  * const __restrict__ dpInteractionLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    T_Flags             const & axis      ,
    Method		        const & met       

){
    auto const dx = DXTableNN_d[ axis ];   // 0 or 1 for  -1,1 
    auto const dy = DYTableNN_d[ axis ];   // 0 or 1 for  -1,1 
    auto const dz = DZTableNN_d[ axis ];   // 0 or 1 for  -1,1 

    auto const x0MTwo = met.getCurve().linearizeBoxVectorIndexX( x0 + dx - uint32_t(2) );
    auto const x0MOne = met.getCurve().linearizeBoxVectorIndexX( x0 + dx - uint32_t(1) );
    auto const x0Abs  = met.getCurve().linearizeBoxVectorIndexX( x0 + dx               );
    auto const x0POne = met.getCurve().linearizeBoxVectorIndexX( x0 + dx + uint32_t(1) );
    auto const x0PTwo = met.getCurve().linearizeBoxVectorIndexX( x0 + dx + uint32_t(2) );

    auto const y0MTwo = met.getCurve().linearizeBoxVectorIndexY( y0 + dy - uint32_t(2) );
    auto const y0MOne = met.getCurve().linearizeBoxVectorIndexY( y0 + dy - uint32_t(1) );
    auto const y0Abs  = met.getCurve().linearizeBoxVectorIndexY( y0 + dy               );
    auto const y0POne = met.getCurve().linearizeBoxVectorIndexY( y0 + dy + uint32_t(1) );
    auto const y0PTwo = met.getCurve().linearizeBoxVectorIndexY( y0 + dy + uint32_t(2) );

    auto const z0MTwo = met.getCurve().linearizeBoxVectorIndexZ( z0 + dz - uint32_t(2) );
    auto const z0MOne = met.getCurve().linearizeBoxVectorIndexZ( z0 + dz - uint32_t(1) );
    auto const z0Abs  = met.getCurve().linearizeBoxVectorIndexZ( z0 + dz               );
    auto const z0POne = met.getCurve().linearizeBoxVectorIndexZ( z0 + dz + uint32_t(1) );
    auto const z0PTwo = met.getCurve().linearizeBoxVectorIndexZ( z0 + dz + uint32_t(2) );

    auto typeA(dpInteractionLattice[ x0Abs + y0Abs + z0Abs ) ]);
    double prop(0);
    switch ( axis >> 1 ){
        case 0 : //+-x
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0POne ]);

            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);
        
            prop*=(dx==0)? (-1) : 1;
            break;
        case 1 : //+-y
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0POne ]);

            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);
                
            prop*=(dy==0)? (-1) : 1;                              
            break;
        case 2 : //+-z
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MTwo ]);
         
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0MOne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
        
            prop*=(dz==0)? (-1) : 1;
            break;
        //TODO : Add diagonal moves 
    }
    return prop;
}


/**
 * @brief add interaction to the species movements
 * 
 * @tparam T_UCoordinateCuda 
 * @tparam moveSize 
 * @param dpPolymerSystem 
 * @param dpPolymerFlags 
 * @param iOffset 
 * @param dpLatticeTmp 
 * @param nMonomers 
 * @param rSeed 
 * @param rGlobalIteration 
 * @param met 
 * @return void 
 */
 template< typename T_UCoordinateCuda >
 __global__ void kernelSimulationScBFMCheckSpeciesInteraction
 (
    T_InteractionTag   * const __restrict__ dpInteractionLattice,
     typename CudaVec4< T_UCoordinateCuda >::value_type
                 const * const __restrict__ dpPolymerSystem         ,
     T_Flags           * const              dpPolymerFlags          ,
     uint32_t            const              iOffset                 ,
     T_Id                const              nMonomers               ,
     uint64_t            const              rSeed                   ,
     uint64_t            const              rGlobalIteration        ,
     Method              const              met
 ){
     uint32_t rn;
     double rnd;
     for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
           iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x ){
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(32) ) == T_Flags(0) ) // impossible move
            continue;

        Saru rng(rGlobalIteration,iMonomer+iOffset,rSeed);
        auto const r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = properties & T_Flags(31); // 7=0b111 31=0b11111
         
        if ( ! ( calcInteractionProbability( dpInteractionLattice, r0.x, r0.y, r0.z, direction, met ) > rng.rng_d() ) ) {
             /* move is not allowed due to the interaction  */
             typename CudaVec4< T_UCoordinateCuda >::value_type const r1 = {
                T_UCoordinateCuda( r0.x + DXTable_d[ direction ] ),
                T_UCoordinateCuda( r0.y + DYTable_d[ direction ] ),
                T_UCoordinateCuda( r0.z + DZTable_d[ direction ] )
            };
             direction &= T_Flags(32) /* cannot -move-modification */;
        }
        dpPolymerFlags[ iMonomer ] = direction;
     }
 }

 template< typename T_UCoordinateCuda> 
 void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_CheckSpeciesInteraction(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, const uint64_t seed)
 {
    kernelSimulationScBFMCheckSpeciesInteraction< T_UCoordinateCuda > 
    <<< nBlocks, nThreads, 0, mStream >>>(     
    mLatticeInteractionTag->gpu,           
    mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],                                     
    mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],           
    mviSubGroupOffsets[ iSpecies ],                                
    mnElementsInGroup[ iSpecies ],                                 
    seed, 
    hGlobalIterator,                                         
    met
    );
   hGlobalIterator++;
 }

template< typename T_UCoordinateCuda >
__global__ void kernelApplyInteraction
T_InteractionTag  * const __restrict__ dpInteractionLattice,
typename CudaVec4< T_UCoordinateCuda >::value_type
            const * const __restrict__ dpPolymerSystem         ,
T_Flags           * const              dpPolymerFlags          ,
uint32_t            const              iOffset                 ,
T_Id                const              nMonomers               ,
uint64_t            const              rSeed                   ,
uint64_t            const              rGlobalIteration        ,
Method              const              met
){
    for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < nMonomers; i += gridDim.x * blockDim.x ){
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ! ( properties & T_Flags(32) ) ) // check if can-move flag is set
            continue; 
        auto const direction = properties & T_Flags(31); // 7=0b111 31=0b11111
        auto const r0 = dpPolymerSystem[ iMonomer ];

        auto const dx = DXTableNN_d[ axis ];   // 0 or 1 for  -1,1 
        auto const dy = DYTableNN_d[ axis ];   // 0 or 1 for  -1,1 
        auto const dz = DZTableNN_d[ axis ];   // 0 or 1 for  -1,1 
    
        auto const x0MOne = met.getCurve().linearizeBoxVectorIndexX( r0.x - uint32_t(1) );
        auto const x0Abs  = met.getCurve().linearizeBoxVectorIndexX( r0.x               );
        auto const x0POne = met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1) );
        auto const x0PTwo = met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(2) );
    
        auto const y0MOne = met.getCurve().linearizeBoxVectorIndexY( r0.y - uint32_t(1) );
        auto const y0Abs  = met.getCurve().linearizeBoxVectorIndexY( r0.y               );
        auto const y0POne = met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(1) );
        auto const y0PTwo = met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(2) );
    
        auto const z0MOne = met.getCurve().linearizeBoxVectorIndexZ( r0.z - uint32_t(1) );
        auto const z0Abs  = met.getCurve().linearizeBoxVectorIndexZ( r0.z               );
        auto const z0POne = met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(1) );
        auto const z0PTwo = met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(2) );

        auto nnTag1(dpInteractionLattice[ x0Abs  ,y0Abs  , z0Abs   ) ]);
        auto nnTag2(T_InteractionTag(0));
        
        switch(direction){ 
            case 0: 
                dpInteractionLattice[ x0MOne + y0Abs  + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0MOne + y0POne + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0MOne + y0Abs  + z0POne ) ] = nnTag1;
                dpInteractionLattice[ x0MOne + y0POne + z0POne ) ] = nnTag1;
                
                dpInteractionLattice[ x0POne + y0Abs  + z0Abs ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0POne + z0Abs ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0Abs  + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0POne + z0POne ) ] = nnTag2;
                break;
            case 1: 
                dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ) ] = nnTag1;
                dpInteractionLattice[ x0PTwo + y0POne + z0POne ) ] = nnTag1;

                dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0POne + z0POne ) ] = nnTag2;
                break;
            case 2: 
                dpInteractionLattice[ x0Abs  + y0MOne + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0POne + y0MOne + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0Abs  + y0MOne + z0POne ) ] = nnTag1;
                dpInteractionLattice[ x0POne + y0MOne + z0POne ) ] = nnTag1;
                
                dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0POne + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0POne + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0POne + z0POne ) ] = nnTag2;
                break;
            case 3: 
                dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0PTwo + y0PTwo + z0Abs  ) ] = nnTag1;
                dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ) ] = nnTag1;
                dpInteractionLattice[ x0PTwo + y0PTwo + z0POne ) ] = nnTag1;

                dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0Abs  + z0POne ) ] = nnTag2;
                break;
            case 4: 
                dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ) ] = nnTag1;
                dpInteractionLattice[ x0Abs  + y0POne + z0MOne ) ] = nnTag1;
                dpInteractionLattice[ x0POne + y0Abs  + z0MOne ) ] = nnTag1;
                dpInteractionLattice[ x0POne + y0POne + z0MOne ) ] = nnTag1;
                
                dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0POne + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0Abs  + z0POne ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0POne + z0POne ) ] = nnTag2;
                break;
            case 5: 
                dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ) ] = nnTag1;
                dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ) ] = nnTag1;
                dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ) ] = nnTag1;
                dpInteractionLattice[ x0POne + y0POne + z0PTwo ) ] = nnTag1;

                dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ) ] = nnTag2;
                dpInteractionLattice[ x0POne + y0POne + z0Abs  ) ] = nnTag2;
                break;
        }
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::launch_ApplyInteraction(
  const size_t nBlocks , const size_t   nThreads, const size_t iSpecies
){ 
	kernelApplyConnection<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
        mLatticeInteractionTag->gpu,           
        mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],                                     
        mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],           
        mviSubGroupOffsets[ iSpecies ],                                
        mnElementsInGroup[ iSpecies ],                                 
        seed, 
        hGlobalIterator,                                         
        met
	);
}
///////////////////////////////////////////////////////////////////////////////
//Define othe member functions/////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda > 
UpdaterGPU_Interaction<T_UCoordinateCuda>::UpdaterGPU_Interaction():
BaseClass 				(	   ),         
mLatticeInteractionTag  ( NULL ),
mInteractionTag        	( NULL )
{
    /**
     * Log control.
     * Note that "Check" controls not the output, but the actualy checks
     * If a checks needs to always be done, then do that check and declare
     * the output as "Info" log level
     */
    mLog.file( __FILENAME__ );
    mLog.deactivate( "Check"     );
    mLog.deactivate( "Error"     );
    mLog.deactivate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
    for(size_t n=0;n<256;n++){
      	for(size_t m=0;m<256;m++){	
			interactionTable[m][n]=0.0;
			probabilityLookup[m][n]=1.0;
        }
    }
};
template< typename T_UCoordinateCuda > 
void UpdaterGPU_Interaction<T_UCoordinateCuda>::destruct(){
	DeleteMirroredObject deletePointer;
	deletePointer( mLatticeInteractionTag, "mLatticeInteractionTag");
    deletePointer(        mInteractionTag,        "mInteractionTag");
	if ( deletePointer.nBytesFreed > 0 ){
		mLog( "Info" )
			<< "Freed a total of "
			<< prettyPrintBytes( deletePointer.nBytesFreed )
			<< " on GPU and host RAM.\n";
	}
}
template< typename T_UCoordinateCuda > 
UpdaterGPU_Interaction<T_UCoordinateCuda>::~UpdaterGPU_Interaction(){
	this->destruct();    
	destruct();
}
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::cleanup(){
    this->destruct();    
    destruct();
    cudaDeviceSynchronize();
    cudaProfilerStop();
}
template < typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::initialize(){
	BaseClass::setAutoColoring(false);
	// mLog( "Info" )<< "Start manual coloring of the graph...\n" ;
	// AStarSpecies = 0; 
	// BStarSpecies = 1; 
	// StarCenterSpecies = 2; 
	// //do manual coloring 
	// for ( uint32_t i = 0; i < mnAllMonomers ; i++){
	// 	T_Id color(( i % 2)==0 ? 3 :4);
	// 	if ( mNeighbors->host[ i ].size == nBranches ) color=2;
	// 	mGroupIds.push_back(color); 

	// }
	// uint32_t nAMonomers(nAStars*(1+nBranches*branchLenghtA) );
	// for (uint32_t i = 0; i < nReactiveMonomers; i++){
	// 	mGroupIds[mNewToOldReactiveID[i]] = (mNewToOldReactiveID[i] < nAMonomers ) ? AStarSpecies : BStarSpecies ;
	// 	if (i <20 ) 
	// 	mLog( "Info" )<< "mGroups[" << mNewToOldReactiveID[i] << "]= "<< mGroupIds[mNewToOldReactiveID[i]] <<"\n" ;
	// }
	// mLog( "Info" )<< "Start manual coloring of the graph...done\n" ;

	mLog( "Info" )<< "Initialize baseclass \n" ;
	BaseClass::initialize();
	{ decltype( dcBoxX  ) x = mBoxX  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX  , &x, sizeof(x) ) ); }
	{ decltype( dcBoxY  ) x = mBoxY  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY  , &x, sizeof(x) ) ); }
	{ decltype( dcBoxZ  ) x = mBoxZ  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ  , &x, sizeof(x) ) ); }
	{ decltype( dcBoxXM1) x = mBoxXM1; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1, &x, sizeof(x) ) ); }
	{ decltype( dcBoxYM1) x = mBoxYM1; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1, &x, sizeof(x) ) ); }
	{ decltype( dcBoxZM1) x = mBoxZM1; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1, &x, sizeof(x) ) ); }
	uint32_t tmp_DXTableNN[18] = {  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint32_t tmp_DYTableNN[18] = {  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint32_t tmp_DZTableNN[18] = {  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	CUDA_ERROR( cudaMemcpyToSymbol( DXTableNN_d, tmp_DXTableNN, sizeof( tmp_DXTableNN ) ) ); 
	CUDA_ERROR( cudaMemcpyToSymbol( DYTableNN_d, tmp_DYTableNN, sizeof( tmp_DXTableNN ) ) );
	CUDA_ERROR( cudaMemcpyToSymbol( DZTableNN_d, tmp_DZTableNN, sizeof( tmp_DXTableNN ) ) );
	mLog( "Info" )<< "Initialize baseclass.done. \n" ;	

	initializeInteractionLattice();
	mLog( "Info" )<< "Initialize lattice.done. \n" ;
    for (auto i=0; i<20; i++ )
        for (auto j=0; j<20; j++ )
            mLog( "Info" )<< "interaction: probabilityLookup[" <<  i  <<","<<j << "]="<< probabilityLookup[i+1][j+1]  <<"\n";
    CUDA_ERROR( cudaMemcpyToSymbol( dcNNProbability, probabilityLookup, sizeof(probabilityLookup) );
	miNewToi->popAsync();
	CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
}
////////////////////////////////////////////////////////////////////////////////
//implement setter function for the interaction tags and their energy //////////
////////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::setInteractionTag(uint32_t id, 
    uint8_t tag ){setAttributeTag(id, tag);}

template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::setNNInteraction(int32_t typeA, 
    int32_t typeB, double energy){
    if(0<typeA && typeA<=255 && 0<typeB && typeB<=255){
        interactionTable[typeA+1][typeB+1]=energy;
        interactionTable[typeB+1][typeA+1]=energy;
        probabilityLookup[typeA+1][typeB+1]=exp(-energy);
        probabilityLookup[typeB+1][typeA+1]=exp(-energy);
        std::cout<<"set interation between types ";
        std::cout<<typeA<<" and "<<typeB<<" to "<<energy<<"kT\n";
    } else {
        std::stringstream errormessage;
        errormessage<<"UpdaterGPU_Interaction::setNNInteraction(typeA,typeB,energy).\n";
        errormessage<<"typeA "<<typeA<<" typeB "<<typeB<<": Types out of range\n";
        throw std::runtime_error(errormessage.str());
    }
}

template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::getNNInteraction(int32_t typeA, 
    int32_t typeB) const{
    if(0<typeA && typeA<=255 && 0<typeB && typeB<=255)
        return interactionTable[typeA+1][typeB+1];
    else{
        std::stringstream errormessage;
        errormessage<<"UpdaterGPU_Interaction::getNNInteraction(typeA,typeB).\n";
        errormessage<<"typeA "<<typeA<<" typeB "<<typeB<<": Types out of range\n";
        throw std::runtime_error(errormessage.str());
    }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda  >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::runSimulationOnGPU( 
    uint32_t const nMonteCarloSteps ){
    std::clock_t const t0 = std::clock();
    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
    CUDA_ERROR( cudaMemcpy( mPolymerSystemSortedOld->gpu, 
                            mPolymerSystemSorted->gpu, 
                            mPolymerSystemSortedOld->nBytes, 
                            cudaMemcpyDeviceToDevice ) );
    auto const nSpecies = mnElementsInGroup.size();
    AutomaticThreadChooser chooseThreads(nSpecies);
    chooseThreads.initialize(mCudaProps);
    std::vector< uint64_t > nSpeciesChosen( nSpecies ,0 );
    /* run simulation */
    for ( uint32_t iStep = 0; iStep < nMonteCarloSteps; ++iStep, ++mAge ){
        if ( useOverflowChecks ){
            /**
             * for uint8_t we have to check for overflows every 127 steps, as
             * for 128 steps we couldn't say whether it actually moved 128 steps
             * or whether it moved 128 steps in the other direction and was wrapped
             * to be equal to the hypothetical monomer above
             */
            auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
            auto constexpr nStepsBetweenOverflowChecks = boxSizeCudaType / 2 - 1;
            if ( iStep != 0 && iStep % nStepsBetweenOverflowChecks == 0 ){
                findAndRemoveOverflows( false );
                CUDA_ERROR( cudaMemcpyAsync( mPolymerSystemSortedOld->gpu,
                    mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes,
                    cudaMemcpyDeviceToDevice, mStream ) );
            }
        }
        /* one Monte-Carlo step:
         *  - tries to move on average all particles one time
         *  - each particle could be touched, not just one group */
        for ( uint32_t iSubStep = 0; iSubStep < nSpecies; ++iSubStep ) 
		{
            auto const iStepTotal = iStep * nSpecies + iSubStep;
            auto  iOffsetLatticeTmp = ( iStepTotal % mnLatticeTmpBuffers )
            * ( mBoxX * mBoxY * mBoxZ * sizeof( mLatticeTmp->gpu[0] ));
            if (met.getPacking().getBitPackingOn()) 
                iOffsetLatticeTmp /= CHAR_BIT;
            auto texLatticeTmp = mvtLatticeTmp[ iStepTotal % mnLatticeTmpBuffers ];
            if (met.getPacking().getNBufferedTmpLatticeOn()) {
                    iOffsetLatticeTmp = 0u;
                    texLatticeTmp = mLatticeTmp->texture;
            }
            /* randomly choose which monomer group to advance */
            auto const iSpecies = randomNumbers.r250_rand32() % nSpecies;
            auto const seed     = randomNumbers.r250_rand32();
            auto const nThreads = chooseThreads.getBestThread(iSpecies);
            auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
            auto const useCudaMemset = chooseThreads.useCudaMemset(iSpecies);
            chooseThreads.addRecord(iSpecies, mStream);

            nSpeciesChosen[ iSpecies ] += 1;
			// if (!diagMovesOn) {
				this-> template launch_CheckSpecies<6>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
                launch_CheckSpeciesInteraction(nBlocks, nThreads, iSpecies,seed );
				if ( useCudaMemset )
					launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
				else
					launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp );
                launch_ApplyInteraction(nBlocks, nThreads, iSpecies);
			// }else{
			// 	this-> template launch_CheckSpecies<18>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
			// 	if ( useCudaMemset )
			// 		launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
			// 	else
			// 		launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp );
			// }
            if ( useCudaMemset ){
                if(met.getPacking().getNBufferedTmpLatticeOn()){
                    /* we only need to delete when buffers will wrap around and
                        * on the last loop, so that on next runSimulationOnGPU
                        * call mLatticeTmp is clean */
                    if ( ( iStepTotal % mnLatticeTmpBuffers == 0 ) ||
                        ( iStep == nMonteCarloSteps-1 && iSubStep == nSpecies-1 ) )
                    {
                        cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
                    }
                }else
                    mLatticeTmp->memsetAsync(0);
            }
            else
                launch_ZeroArraySpecies(nBlocks,nThreads,iSpecies);
            chooseThreads.analyze(iSpecies,mStream);
		} // iSubstep
    } // iStep
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
        << "run time (GPU): " << nMonteCarloSteps << "\n"
        << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
        << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
	checkSystem(); // no-op if "Check"-level deactivated
}
template class UpdaterGPU_Interaction< uint8_t  >;
template class UpdaterGPU_Interaction< uint16_t >;
template class UpdaterGPU_Interaction< uint32_t >;
template class UpdaterGPU_Interaction<  int16_t >;
// template class UpdaterGPU_Interaction<  int32_t >;