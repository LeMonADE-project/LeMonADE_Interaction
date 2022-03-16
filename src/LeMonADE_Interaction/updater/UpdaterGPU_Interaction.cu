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
#include <LeMonADEGPU/utility/graphColoring.h>
#include <LeMonADEGPU/core/rngs/Saru.h>
#include <LeMonADEGPU/core/MonomerEdges.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/feature/BoxCheck.h>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <LeMonADEGPU/core/BondVectorSet.h>
///////////////////////////////////////////////////////////////////////////////
using T_Flags            = UpdaterGPU_Interaction< uint8_t >::T_Flags         ;
using T_Id               = UpdaterGPU_Interaction< uint8_t >::T_Id            ;
using T_InteractionTag   = UpdaterGPU_Interaction< uint8_t >::T_InteractionTag;
using T_Color            = UpdaterGPU_Interaction< uint8_t >::T_Color         ;
///////////////////////////////////////////////////////////////////////////////
__device__ __constant__ uint32_t DXTableNN_d[18];
__device__ __constant__ uint32_t DYTableNN_d[18];
__device__ __constant__ uint32_t DZTableNN_d[18];
__device__ __constant__ double dcNNProbability[32][32];
///////////////////////////////////////////////////////////////////////////////
__global__ void  kernelPrintTagType(){
    printf("TagType[%d][%d]=%f\n",blockIdx.x,  threadIdx.x, dcNNProbability[blockIdx.x][threadIdx.x] );
}
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
        T_InteractionTag const interactionTag( dInteractionTag[iOffset + id] + T_InteractionTag(1) );

        auto const x0Abs  = met.getCurve().linearizeBoxVectorIndexX( r0.x               );
        auto const x0POne = met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1) );
    
        auto const y0Abs  = met.getCurve().linearizeBoxVectorIndexY( r0.y                );
        auto const y0POne = met.getCurve().linearizeBoxVectorIndexY( r0.y  + uint32_t(1) );
    
        auto const z0Abs  = met.getCurve().linearizeBoxVectorIndexZ( r0.z                );
        auto const z0POne = met.getCurve().linearizeBoxVectorIndexZ( r0.z  + uint32_t(1) );
        
        if (
            dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0Abs  + y0POne + z0POne ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0POne + y0POne + z0Abs  ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0POne + y0Abs  + z0POne ] != T_InteractionTag(0) ||
            dpInteractionLattice[ x0POne + y0POne + z0POne ] != T_InteractionTag(0) 
        ) {
            printf("Occupy an already occupied lattice site: %d %d %d %d %d %d %d %d\n",   
            dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ] ,
            dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ] ,
            dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ] ,
            dpInteractionLattice[ x0Abs  + y0POne + z0POne ] ,
            dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ] ,
            dpInteractionLattice[ x0POne + y0POne + z0Abs  ] ,
            dpInteractionLattice[ x0POne + y0Abs  + z0POne ] ,
            dpInteractionLattice[ x0POne + y0POne + z0POne ] );
        }
        dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ] = interactionTag;
        dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ] = interactionTag;
        dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ] = interactionTag;
        dpInteractionLattice[ x0Abs  + y0POne + z0POne ] = interactionTag;
        dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ] = interactionTag;
        dpInteractionLattice[ x0POne + y0POne + z0Abs  ] = interactionTag;
        dpInteractionLattice[ x0POne + y0Abs  + z0POne ] = interactionTag;
        dpInteractionLattice[ x0POne + y0POne + z0POne ] = interactionTag;
    }
}
 /**
  * @brief convinience function to update the lattice occupation. 
  * @details We introduce such functions because then they can be used latter on from inheritate classes..
  */
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::launch_initializeInteractionLattice(
  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies ){
	// mLog ( "Check" ) <<"Start filling lattice with ones:  \n" ;
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
        T_InteractionTag const interactionTagReset(0);
        T_InteractionTag const interactionTag( dInteractionTag[iOffset + id] + T_InteractionTag(1) );

        auto const x0Abs  = met.getCurve().linearizeBoxVectorIndexX( r0.x               );
        auto const x0POne = met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1) );
    
        auto const y0Abs  = met.getCurve().linearizeBoxVectorIndexY( r0.y                );
        auto const y0POne = met.getCurve().linearizeBoxVectorIndexY( r0.y  + uint32_t(1) );
    
        auto const z0Abs  = met.getCurve().linearizeBoxVectorIndexZ( r0.z                );
        auto const z0POne = met.getCurve().linearizeBoxVectorIndexZ( r0.z  + uint32_t(1) );
        if (
            dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ] != interactionTag ||
            dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ] != interactionTag ||
            dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ] != interactionTag ||
            dpInteractionLattice[ x0Abs  + y0POne + z0POne ] != interactionTag ||
            dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ] != interactionTag ||
            dpInteractionLattice[ x0POne + y0POne + z0Abs  ] != interactionTag ||
            dpInteractionLattice[ x0POne + y0Abs  + z0POne ] != interactionTag ||
            dpInteractionLattice[ x0POne + y0POne + z0POne ] != interactionTag 
        ) {
            printf("Occupy an already occupied lattice site: %d %d %d %d %d %d %d %d\n",   
            dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ] ,
            dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ] ,
            dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ] ,
            dpInteractionLattice[ x0Abs  + y0POne + z0POne ] ,
            dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ] ,
            dpInteractionLattice[ x0POne + y0POne + z0Abs  ] ,
            dpInteractionLattice[ x0POne + y0Abs  + z0POne ] ,
            dpInteractionLattice[ x0POne + y0POne + z0POne ] );
        }
        dpInteractionLattice[ x0Abs  + y0Abs  + z0Abs  ] = interactionTagReset;
        dpInteractionLattice[ x0Abs  + y0POne + z0Abs  ] = interactionTagReset;
        dpInteractionLattice[ x0Abs  + y0Abs  + z0POne ] = interactionTagReset;
        dpInteractionLattice[ x0Abs  + y0POne + z0POne ] = interactionTagReset;
        dpInteractionLattice[ x0POne + y0Abs  + z0Abs  ] = interactionTagReset;
        dpInteractionLattice[ x0POne + y0POne + z0Abs  ] = interactionTagReset;
        dpInteractionLattice[ x0POne + y0Abs  + z0POne ] = interactionTagReset;
        dpInteractionLattice[ x0POne + y0POne + z0POne ] = interactionTagReset;
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
        mInteractionTag->gpu          ,             
		mviSubGroupOffsets[ iSpecies ], 
		mLatticeInteractionTag->gpu   ,
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
    mPolymerSystemSorted->pop(0);
    miToiNew->pop(0);
    CUDA_ERROR( cudaStreamSynchronize( mStream ) );
	uint32_t countLatticeEntries(0);
    uint32_t countLatticeEntriesAType(0);
    uint32_t countLatticeEntriesBType(0);
	for(T_Id x=0; x< mBoxX; x++ )
		for(T_Id y=0; y< mBoxY; y++ )
			for(T_Id z=0; z< mBoxX; z++ ){
                uint32_t tag(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)]);
				if( tag > 0 ){
					countLatticeEntries++;
                    if (tag == 3 )countLatticeEntriesAType++;
                    if (tag == 4 )countLatticeEntriesBType++;

                }
            }
	assert(mnAllMonomers*8 == countLatticeEntries );  
    assert(mnAllMonomers*4 == countLatticeEntriesAType );  
    assert(mnAllMonomers*4 == countLatticeEntriesBType );  
	// mLog( "Check" )
    std::cout 
		<< "checkInteractionLatticeOccupation: \n"
        << "mnAllMonomers*8          = " <<     mnAllMonomers*8 << "\n"
        << "mnAllMonomers*4          = " <<     mnAllMonomers*4 << "\n"
		<< "mnAllMonomers            = " <<       mnAllMonomers << "\n"
		<< "countLatticeEntries      = " << countLatticeEntries << "\n"
        << "countLatticeEntriesAType = " << countLatticeEntriesAType << "\n"
        << "countLatticeEntriesBType = " << countLatticeEntriesBType << std::endl;
    //TODO:teseting the consistency of the lattice and the specied position
	
    for (auto i=0; i < mnAllMonomers; i++){
        auto const r0(mPolymerSystemSorted->host[miToiNew->host[i]]);
        uint32_t x=r0.x;
        uint32_t y=r0.y;
        uint32_t z=r0.z;
        if (
            !(
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y  ,z  )]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y+1,z  )]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y  ,z+1)]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y+1,z+1)]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y  ,z  )]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y+1,z  )]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y  ,z+1)]) &&
            (getAttributeTag(i)+1)== static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y+1,z+1)]) 
            )
        ){
            std::stringstream error_message;
            error_message << "AttributeTag["<<i<<"]="<<getAttributeTag(i)+1<<"\n";
            error_message << "LatticeEntry["<<x  <<","<<y  <<","<<z  <<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y  ,z  )])<< "\n"; 
            error_message << "LatticeEntry["<<x  <<","<<y+1<<","<<z  <<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y+1,z  )])<< "\n"; 
            error_message << "LatticeEntry["<<x  <<","<<y  <<","<<z+1<<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y  ,z+1)])<< "\n"; 
            error_message << "LatticeEntry["<<x  <<","<<y+1<<","<<z+1<<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x  ,y+1,z+1)])<< "\n"; 
            error_message << "LatticeEntry["<<x+1<<","<<y  <<","<<z  <<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y  ,z  )])<< "\n"; 
            error_message << "LatticeEntry["<<x+1<<","<<y+1<<","<<z  <<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y+1,z  )])<< "\n"; 
            error_message << "LatticeEntry["<<x+1<<","<<y  <<","<<z+1<<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y  ,z+1)])<< "\n"; 
            error_message << "LatticeEntry["<<x+1<<","<<y+1<<","<<z+1<<"]="<< static_cast<uint32_t>(mLatticeInteractionTag->host[met.getCurve().linearizeBoxVectorIndex(x+1,y+1,z+1)])<< "\n"; 
            throw std::runtime_error(error_message.str());
        }
    }
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

    auto const typeA(dpInteractionLattice[ x0Abs + y0Abs + z0Abs ] );
    double prop(1);
    uint32_t newPositions[16];
    uint32_t oldPositions[16];
    switch ( axis >> 1 ){
        case 0 : //+-x
        
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);

            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0POne ]);
        
            if(dx==0){prop=1./prop;}
            break;
        case 1 : //+-y
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0POne ]);

            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);           
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0PTwo ]);

            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);


            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);
                
            if(dy==0){prop=1./prop;}                        
            break;
        case 2 : //+-z
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MTwo ]);
         
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0MOne ]);

            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);

            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
        
            if(dz==0){prop=1./prop;}
            break;
        //TODO : Add diagonal moves 
        case 3: 
            //+x+y 
            //-x-y 
            //new positions
            //right
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);
            
            //top
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);

            //front
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MOne ]);
            
            //back
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
            

            //old positions 
            //left
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0MOne + z0POne ]);
            
            //bottom
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MTwo + z0POne ]);

            //front
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0MOne ]);
            
            //back
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0PTwo ]);
            if (dx == 0 ){prop=1./prop;}
            break;
        case 4: 
            //+x-y 
            //-x+y 
            //new positions
            //right
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0POne ]);
            //bottom
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0POne ]);
            //front
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MOne ]);
            //back
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0PTwo ]);

            //old positions 
            //left
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0POne ]);
            //top
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0Abs  ]);
            //front
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            //back
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0PTwo ]);
            
            if (dx == 1 ){prop=1./prop;}
            break;
        case 5:
            //+y+z 
            //-y-z 
            //new positions
            //top
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);
            //right
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
            
            //front
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0POne ]);
            
            //back
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);            

            //old positions 
            //bottom
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0MOne ]);

            //left
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MTwo ]);

            //front
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0MOne ]);
            
            //back
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0MOne ]);
            
            if (dy == 0 ){prop=1./prop;}
            break;
        case 6: 
            //+y-z 
            //-y+z 
            //new positions
            //top
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0MOne ]);
            //right
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MTwo ]);
            //front
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0POne ]);
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            //back
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);

            //old positions 
            //top
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MTwo + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MTwo + z0POne ]);
            //left
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0PTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            //front
            // prop/=getProbability(typeA, dpInteractionLattice[ x0MOne  + y0Abs  + z0Abs ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0POne ]);
            //back
            // prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo  + y0Abs  + z0Abs ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0MOne + z0POne ]);
            
            if (dy == 1 ){prop=1./prop;}
            break;
        case 7:
            //+z+x
            //-z-x 
            //new positions
            //right
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0PTwo ]);
            //top
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0POne ]);
            //front
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0POne   ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0POne ]);
            //back
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0POne ]);
            

            //old positions 
            //left
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MTwo ]);
            //bottom
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0POne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0POne ]);
            //front
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne  + z0Abs ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0MOne ]);
            
            //back
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo  + z0Abs ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0MOne ]);
            
            if (dz == 0 ){prop=1./prop;}
            break;
        case 8: 
            //+z-x
            //-z+x 
            //new positions
            //right
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0Abs  + z0PTwo ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0POne + z0PTwo ]);
            //bottom
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0Abs  + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MTwo + y0POne + z0POne ]);
            //front
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MOne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0MOne + z0POne ]);
            //back
            // prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo  + z0Abs ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0Abs  ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0POne ]);
            prop*=getProbability(typeA, dpInteractionLattice[ x0MOne + y0PTwo + z0POne ]);
            
            //old positions 
            //top
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0POne + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0Abs  + z0MTwo ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0POne + z0MTwo ]);
            //left
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0Abs  + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0PTwo + y0POne + z0MOne ]);            
            //front
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne  + z0Abs ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0MOne + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0MOne + z0MOne ]);
            //back
            // prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo  + z0Abs ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0Abs  + y0PTwo + z0MOne ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0Abs  ]);
            prop/=getProbability(typeA, dpInteractionLattice[ x0POne + y0PTwo + z0MOne ]);
            
            if (dz == 1 ){prop=1./prop;}
            break;
    }
    return prop;
}
/**
 * @brief add interaction to the species movements
 * 
 * @tparam T_UCoordinateCuda 
 * @param dpPolymerSystem 
 * @param dpPolymerFlags 
 * @param iOffset 
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
    for ( T_Id id = blockIdx.x * blockDim.x + threadIdx.x;
        id < nMonomers; id += gridDim.x * blockDim.x ){
        auto const properties = dpPolymerFlags[ id ];
        if ( ( properties & T_Flags(32) ) == T_Flags(0) ) // impossible move
            continue;

        auto direction = properties & T_Flags(31); // 7=0b111 31=0b11111
        auto const r0 = dpPolymerSystem[ id ];
        auto const intProp(calcInteractionProbability( dpInteractionLattice, r0.x, r0.y, r0.z, direction, met ));
        Saru rng(rGlobalIteration,id+iOffset,rSeed);
        if ( rng.rng_d() < intProp ) {
             /* move is still allowed */
            direction += T_Flags(32);
        }
        dpPolymerFlags[ id ] = direction;
     }
 }

 template< typename T_UCoordinateCuda> 
 void UpdaterGPU_Interaction< T_UCoordinateCuda >::launch_CheckSpeciesInteraction(
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
__global__ void kernelApplyInteraction(
T_InteractionTag  * const __restrict__ dpInteractionLattice    ,
typename CudaVec4< T_UCoordinateCuda >::value_type
            const * const __restrict__ dpPolymerSystem         ,
T_Flags           * const              dpPolymerFlags          ,
T_Id                const              nMonomers               ,
Method              const              met
){
    for ( T_Id id = blockIdx.x * blockDim.x + threadIdx.x;
        id < nMonomers; id += gridDim.x * blockDim.x ){
        auto const properties = dpPolymerFlags[ id ];
        if ( ! ( properties & T_Flags(32) ) ) // impossible move
            continue; 
        auto const direction = properties & T_Flags(31); // 7=0b111 31=0b11111
        /** The positions are already updated!
         * Therfore, we substract the direction to obtain the old position,
         * which were assumed in the switch-statement. 
         * Problem : The DXTable_d is not set within this file scope!!!
         * Solution: Rewrite the adressing of the lattice...
         */
        auto const r0 = dpPolymerSystem[ id ] ;

        auto const x0MOne = met.getCurve().linearizeBoxVectorIndexX( r0.x - uint32_t(1) );
        auto const x0Abs  = met.getCurve().linearizeBoxVectorIndexX( r0.x               );
        auto const x0POne = met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1) );

        auto const y0MOne = met.getCurve().linearizeBoxVectorIndexY( r0.y - uint32_t(1) );
        auto const y0Abs  = met.getCurve().linearizeBoxVectorIndexY( r0.y               );
        auto const y0POne = met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(1) );

        auto const z0MOne = met.getCurve().linearizeBoxVectorIndexZ( r0.z - uint32_t(1) );
        auto const z0Abs  = met.getCurve().linearizeBoxVectorIndexZ( r0.z               );
        auto const z0POne = met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(1) );

        uint32_t newPositions[6];
        uint32_t oldPositions[6];
        
        auto const dx = DXTableNN_d[ direction ];   // 
        auto const dy = DYTableNN_d[ direction ];   // 
        auto const dz = DZTableNN_d[ direction ];   // 

        switch( direction >> 1 ){ 
            case 0://-x and +x
                newPositions[0]=met.getCurve().linearizeBoxVectorIndexX( r0.x + dx );
                newPositions[2]=newPositions[0] + y0POne;
                newPositions[0]+=y0Abs ;
                newPositions[1]=newPositions[0] + z0Abs;
                newPositions[3]=newPositions[2] + z0POne;
                newPositions[0]+=z0POne;
                newPositions[2]+=z0Abs;

                oldPositions[0]=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(2-2*dx) - uint32_t(dx) );
                oldPositions[2]=oldPositions[0] + y0POne;
                oldPositions[0]+=y0Abs ;
                oldPositions[1]=oldPositions[0] + z0Abs;
                oldPositions[3]=oldPositions[2] + z0POne;
                oldPositions[0]+=z0POne;
                oldPositions[2]+=z0Abs;
                break;
            case 1://-y and +y
                newPositions[0]=met.getCurve().linearizeBoxVectorIndexY( r0.y + dy );
                newPositions[2]=newPositions[0] + z0POne;
                newPositions[0]+=z0Abs ;
                newPositions[1]=newPositions[0] + x0Abs;
                newPositions[3]=newPositions[2] + x0POne;
                newPositions[0]+=x0POne;
                newPositions[2]+=x0Abs;

                oldPositions[0]=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(2-2*dy) - uint32_t(dy) );
                oldPositions[2]=oldPositions[0] + z0POne;
                oldPositions[0]+=z0Abs ;
                oldPositions[1]=oldPositions[0] + x0Abs;
                oldPositions[3]=oldPositions[2] + x0POne;
                oldPositions[0]+=x0POne;
                oldPositions[2]+=x0Abs;
                break;
            case 2://-z and +z
                newPositions[0]=met.getCurve().linearizeBoxVectorIndexZ( r0.z + dz );
                newPositions[2]=newPositions[0] + x0POne;
                newPositions[0]+=x0Abs ;
                newPositions[1]=newPositions[0] + y0Abs;
                newPositions[3]=newPositions[2] + y0POne;
                newPositions[0]+=y0POne;
                newPositions[2]+=y0Abs;

                oldPositions[0]=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(2-2*dz) - uint32_t(dz)  );
                oldPositions[2]=oldPositions[0] + x0POne;
                oldPositions[0]+=x0Abs ;
                oldPositions[1]=oldPositions[0] + y0Abs;
                oldPositions[3]=oldPositions[2] + y0POne;
                oldPositions[0]+=y0POne;
                oldPositions[2]+=y0Abs;
                break;
            case 3: // xy and -x-y
            case 4: // -xy and x-y
                newPositions[0]=met.getCurve().linearizeBoxVectorIndexX( r0.x + dx );
                newPositions[2]=met.getCurve().linearizeBoxVectorIndexY( r0.y + dy );
                newPositions[4]=newPositions[0] + newPositions[2];
                newPositions[0]+=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(1-dy)  );
                newPositions[2]+=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1-dx)  );
                newPositions[1]=newPositions[0] + z0POne;
                newPositions[3]=newPositions[2] + z0POne;
                newPositions[5]=newPositions[4] + z0POne;
                newPositions[0]+=z0Abs;
                newPositions[2]+=z0Abs;
                newPositions[4]+=z0Abs;

                oldPositions[0]=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(2-2*dx) - uint32_t(dx) );
                oldPositions[2]=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(2-2*dy) - uint32_t(dy) );
                oldPositions[4]=oldPositions[0] + oldPositions[2];
                oldPositions[0]+=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(1-dy)  );
                oldPositions[2]+=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1-dx)  );
                oldPositions[1]=oldPositions[0] + z0POne;
                oldPositions[3]=oldPositions[2] + z0POne;
                oldPositions[5]=oldPositions[4] + z0POne;
                oldPositions[0]+=z0Abs;
                oldPositions[2]+=z0Abs;
                oldPositions[4]+=z0Abs;
                break;
            case 5: // yz and -y-z
            case 6: // -yz and y-z
                newPositions[0]=met.getCurve().linearizeBoxVectorIndexY( r0.y + dy );
                newPositions[2]=met.getCurve().linearizeBoxVectorIndexZ( r0.z + dz );
                newPositions[4]=newPositions[0] + newPositions[2];
                newPositions[0]+=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(1-dz)  );
                newPositions[2]+=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(1-dy)  );
                newPositions[1]=newPositions[0] + x0POne;
                newPositions[3]=newPositions[2] + x0POne;
                newPositions[5]=newPositions[4] + x0POne;
                newPositions[0]+=x0Abs;
                newPositions[2]+=x0Abs;
                newPositions[4]+=x0Abs;

                oldPositions[0]=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(2-2*dy) - uint32_t(dy) );
                oldPositions[2]=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(2-2*dz) - uint32_t(dz) );
                oldPositions[4]=oldPositions[0] + oldPositions[2];
                oldPositions[0]+=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(1-dz)  );
                oldPositions[2]+=met.getCurve().linearizeBoxVectorIndexY( r0.y + uint32_t(1-dy)  );
                oldPositions[1]=oldPositions[0] + x0POne;
                oldPositions[3]=oldPositions[2] + x0POne;
                oldPositions[5]=oldPositions[4] + x0POne;
                oldPositions[0]+=x0Abs;
                oldPositions[2]+=x0Abs;
                oldPositions[4]+=x0Abs;
                break;
            case 7: // zx and -z-x
            case 8: // -zx and z-x
                newPositions[0]=met.getCurve().linearizeBoxVectorIndexZ( r0.z + dz );
                newPositions[2]=met.getCurve().linearizeBoxVectorIndexX( r0.x + dx );
                newPositions[4]=newPositions[0] + newPositions[2];
                newPositions[0]+=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1-dx)  );
                newPositions[2]+=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(1-dz)  );
                newPositions[1]=newPositions[0] + y0POne;
                newPositions[3]=newPositions[2] + y0POne;
                newPositions[5]=newPositions[4] + y0POne;
                newPositions[0]+=y0Abs;
                newPositions[2]+=y0Abs;
                newPositions[4]+=y0Abs;

                oldPositions[0]=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(2-2*dz) - uint32_t(dz) );
                oldPositions[2]=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(2-2*dx) - uint32_t(dx) );
                oldPositions[4]=oldPositions[0] + oldPositions[2];
                oldPositions[0]+=met.getCurve().linearizeBoxVectorIndexX( r0.x + uint32_t(1-dx)  );
                oldPositions[2]+=met.getCurve().linearizeBoxVectorIndexZ( r0.z + uint32_t(1-dz)  );
                oldPositions[1]=oldPositions[0] + y0POne;
                oldPositions[3]=oldPositions[2] + y0POne;
                oldPositions[5]=oldPositions[4] + y0POne;
                oldPositions[0]+=y0Abs;
                oldPositions[2]+=y0Abs;
                oldPositions[4]+=y0Abs;
                break;
        }
        T_InteractionTag const nnTag1(dpInteractionLattice[ newPositions[0] ]);
        T_InteractionTag const nnTag2(dpInteractionLattice[ oldPositions[0] ]);
        switch( direction >> 1 ){ 
            case 0: 
            case 1: 
            case 2:
                dpInteractionLattice[ oldPositions[0] ] = nnTag1;
                dpInteractionLattice[ oldPositions[1] ] = nnTag1;
                dpInteractionLattice[ oldPositions[2] ] = nnTag1;
                dpInteractionLattice[ oldPositions[3] ] = nnTag1;
                dpInteractionLattice[ newPositions[0] ] = nnTag2;
                dpInteractionLattice[ newPositions[1] ] = nnTag2;
                dpInteractionLattice[ newPositions[2] ] = nnTag2;
                dpInteractionLattice[ newPositions[3] ] = nnTag2;
                break;
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                dpInteractionLattice[ oldPositions[0] ] = nnTag1;
                dpInteractionLattice[ oldPositions[1] ] = nnTag1;
                dpInteractionLattice[ oldPositions[2] ] = nnTag1;
                dpInteractionLattice[ oldPositions[3] ] = nnTag1;
                dpInteractionLattice[ oldPositions[4] ] = nnTag1;
                dpInteractionLattice[ oldPositions[5] ] = nnTag1;
                dpInteractionLattice[ newPositions[0] ] = nnTag2;
                dpInteractionLattice[ newPositions[1] ] = nnTag2;
                dpInteractionLattice[ newPositions[2] ] = nnTag2;
                dpInteractionLattice[ newPositions[3] ] = nnTag2;
                dpInteractionLattice[ newPositions[4] ] = nnTag2;
                dpInteractionLattice[ newPositions[5] ] = nnTag2;
                break;
        }
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction< T_UCoordinateCuda >::launch_ApplyInteraction(
  const size_t nBlocks , const size_t   nThreads, const size_t iSpecies
){ 
	kernelApplyInteraction<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
        mLatticeInteractionTag->gpu,           
        mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],                                     
        mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],                                         
        mnElementsInGroup[ iSpecies ],                           
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
    mLog.activate( "Error"     );
    mLog.activate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
    for(size_t n=0;n<maxInteractionType;n++){
      	for(size_t m=0;m<maxInteractionType;m++){	
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
    mLog( "Info" )<< "Start manual coloring of the graph...\n" ;
    bool const bUniformColors = true; // setting this to true should yield more performance as the kernels are uniformly utilized
    //map with: key=interactionTag, values=number of Monomers with interaction TAg
    std::map<uint32_t,std::vector<uint32_t> > newToOldNNIDs; 
    std::vector<uint32_t> oldToNewNNIDs(mnAllMonomers,0); 
    for(auto i=0; i < mnAllMonomers; i++){
        newToOldNNIDs[getAttributeTag(i)].push_back(i);
        oldToNewNNIDs[i]=newToOldNNIDs[getAttributeTag(i)].size()-1;
    }
    for(auto i=0; i < 20; i++)
        mLog( "Info" )<< "oldToNewNNIDs["<<i<<"]="<< oldToNewNNIDs[i]<<"\n" ; 
    //vector with the interaction Tags
    std::vector<uint32_t> interactionTags; 
    //offset of the number of monomers with the interaction tag 
    std::vector<uint32_t> interactionTagsOffset; 
    //inteaction tag sorted 
    //interaction Tag are for example : 2 3 7 , which are sorted to 0 1 2
    std::map<uint32_t,uint32_t> interactionTagSorted;
    uint32_t tmpCounter(0);
    for(auto it=newToOldNNIDs.begin(); it!=newToOldNNIDs.end();it++){
        interactionTags.push_back(it->first );
        interactionTagSorted[it->first]=tmpCounter;
        tmpCounter++;
    }
    interactionTagsOffset.push_back(0);
    for(auto i=1; i< interactionTags.size();i++){
        interactionTagsOffset.push_back(interactionTagsOffset[i-1] + newToOldNNIDs.at(interactionTags[i]).size());
    }

    mLog( "Info" )<< "There are "<< interactionTags.size()<<" interaction species.\n" ;
    for (auto i=0; i < interactionTags.size(); i++)
        mLog( "Info" )<< "interaction species type "<< interactionTags[i]<<"->"<<interactionTagSorted[interactionTags[i]]<< " size="<<newToOldNNIDs.at(interactionTags[i]).size() << "\n" ;
    //create a neighboring list which contains only on interaction tag species
    std::vector< std::vector< MonomerEdges > >  mSpeciesNeighbors;
    for (auto i=0; i < interactionTags.size(); i++){
        std::vector< MonomerEdges > neighbors(newToOldNNIDs[interactionTags[i]].size(),MonomerEdges());
        mSpeciesNeighbors.push_back(neighbors);
    }
    for(auto i=0; i < mnAllMonomers; i++){
        auto attribute(interactionTagSorted.at(getAttributeTag(i)));
        auto oldID(i);
        MonomerEdges oldNeighbors(mNeighbors->host[oldID]);
        MonomerEdges newNeighbors;
        newNeighbors.size=0;
        for(auto j=0;j<oldNeighbors.size;j++){
            if (getAttributeTag(oldNeighbors.neighborIds[j]) != getAttributeTag(i) ) continue;
            auto neighborID( oldToNewNNIDs[ oldNeighbors.neighborIds[j] ] );            
            newNeighbors.neighborIds[newNeighbors.size]=neighborID;
            newNeighbors.size++;
        }
        auto newID(oldToNewNNIDs[oldID]);
        if (i <20 )
            std::cout << oldID << " " << newID <<  " " << attribute<<std::endl;
        mSpeciesNeighbors[attribute][newID]=newNeighbors;
    }
    // use the automatic coloring algorithm within one interaction tag species
    std::vector< std::vector< T_Color > > mSpeciesGroupIds;
    for (auto i=0; i < interactionTags.size(); i++){
        mSpeciesGroupIds.push_back(
            graphColoring< std::vector<MonomerEdges> const, T_Id, T_Color >(
                mSpeciesNeighbors[i], 
                newToOldNNIDs.at(interactionTags.at(i)).size(), 
                bUniformColors,
                []( std::vector<MonomerEdges> const & x, T_Id const & i ){ return x[i].size; },
                []( std::vector<MonomerEdges> const & x, T_Id const & i, size_t const & j ){ return x[i].neighborIds[j]; }
            )
        );
    }
    //resort the colors to the initial ids 
    mGroupIds.resize(mnAllMonomers,0);
    // for (auto i=0; i < mnAllMonomers; i++){
    auto colorOffset(0);
    for (auto i=0; i < mSpeciesGroupIds.size(); i++){
        auto attribute(interactionTags[i]);
        std::map<uint32_t,uint32_t> usedColors;
        for(auto j=0; j < mSpeciesGroupIds[i].size(); j++){            
            auto oldID(newToOldNNIDs[attribute][j]);
            mGroupIds[oldID]=mSpeciesGroupIds[i][j]+colorOffset;
            usedColors[mGroupIds[oldID]]++;
        }
        colorOffset+=usedColors.size();
    }
    mLog( "Info" )<< "Colors:\n";
    for(auto i=0; i <20;i++)
		mLog( "Info" )<< "mGroups[" << i << "]= "<< mGroupIds[i] <<"\n" ;

	mLog( "Info" )<< "Start manual coloring of the graph...done\n" ;

	mLog( "Info" )<< "Initialize baseclass \n" ;
	BaseClass::initialize();
    size_t nBytesInteractionTagTmp = mnMonomersPadded* sizeof(T_InteractionTag);
    mLog( "Info" ) << "Allocate "<< nBytesInteractionTagTmp/1024<<"kB  memory for mInteractionTag \n";  
    mInteractionTag  = new MirroredTexture< T_InteractionTag >( nBytesInteractionTagTmp, mStream );
    miToiNew->popAsync();
	CUDA_ERROR( cudaStreamSynchronize( mStream ) );
    for( auto i=0;i<mnAllMonomers; i++)
        mInteractionTag->host[miToiNew->host[i]]=static_cast<uint8_t>(getAttributeTag(i)); 
    mInteractionTag->push(0);
    cudaStreamSynchronize( mStream );


	{ decltype( dcBoxX  ) x = mBoxX  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX  , &x, sizeof(x) ) ); }
	{ decltype( dcBoxY  ) x = mBoxY  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY  , &x, sizeof(x) ) ); }
	{ decltype( dcBoxZ  ) x = mBoxZ  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ  , &x, sizeof(x) ) ); }
	{ decltype( dcBoxXM1) x = mBoxXM1; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1, &x, sizeof(x) ) ); }
	{ decltype( dcBoxYM1) x = mBoxYM1; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1, &x, sizeof(x) ) ); }
	{ decltype( dcBoxZM1) x = mBoxZM1; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1, &x, sizeof(x) ) ); }
    uint64_t mBoxXLog2(0), mBoxXYLog2(0);
    { auto dummy = mBoxX ; while ( dummy >>= 1 ) ++mBoxXLog2;
      dummy = mBoxX*mBoxY; while ( dummy >>= 1 ) ++mBoxXYLog2;}
    { decltype( dcBoxXLog2  ) x = mBoxXLog2  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
    { decltype( dcBoxXYLog2 ) x = mBoxXYLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); } 

	uint32_t tmp_DXTableNN[18] = {  0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0 };
	uint32_t tmp_DYTableNN[18] = {  0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0 };
	uint32_t tmp_DZTableNN[18] = {  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1 };
	CUDA_ERROR( cudaMemcpyToSymbol( DXTableNN_d, tmp_DXTableNN, sizeof( tmp_DXTableNN ) ) ); 
	CUDA_ERROR( cudaMemcpyToSymbol( DYTableNN_d, tmp_DYTableNN, sizeof( tmp_DYTableNN ) ) );
	CUDA_ERROR( cudaMemcpyToSymbol( DZTableNN_d, tmp_DZTableNN, sizeof( tmp_DZTableNN ) ) );
    CUDA_ERROR( cudaStreamSynchronize( mStream ) );
	mLog( "Info" )<< "Initialize baseclass.done. \n" ;	

	initializeInteractionLattice();
    auto const nSpecies = mnElementsInGroup.size();
    for ( uint32_t iSpecies = 0; iSpecies < nSpecies; ++iSpecies ){
        /* randomly choose which monomer group to advance */
        auto const nThreads = 256;
        auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
        launch_initializeInteractionLattice(nBlocks,nThreads,iSpecies);
    }
    checkInteractionLatticeOccupation();
	mLog( "Info" )<< "Initialize lattice.done. \n" ;

    for (auto i=0; i<20; i++ )
        for (auto j=0; j<20; j++ )
            mLog( "Info" )<< "interaction: probabilityLookup[" <<  i  <<","<<j << "]="<< probabilityLookup[i+1][j+1]  <<"\n";
    CUDA_ERROR( cudaMemcpyToSymbol( dcNNProbability, probabilityLookup, sizeof(probabilityLookup) ));
    checkInteractionLatticeOccupation();
    kernelPrintTagType<<<5,5>>>();
    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
}
////////////////////////////////////////////////////////////////////////////////
//implement setter function for the interaction tags and their energy //////////
////////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::setInteractionTag(
    uint32_t id, uint8_t tag ){
    setAttributeTag(id, static_cast<uint32_t>(tag));
}
////////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda >
void UpdaterGPU_Interaction<T_UCoordinateCuda>::setNNInteraction(
    int32_t typeA, int32_t typeB, double energy){
    if(0<typeA && typeA<=maxInteractionType && 0<typeB && typeB<=maxInteractionType){
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
////////////////////////////////////////////////////////////////////////////////
template< typename T_UCoordinateCuda >
double UpdaterGPU_Interaction<T_UCoordinateCuda>::getNNInteraction(int32_t typeA, 
    int32_t typeB) const {
    if(0<typeA && typeA<=maxInteractionType && 0<typeB && typeB<=maxInteractionType)
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
            // auto const useCudaMemset = chooseThreads.useCudaMemset(iSpecies);
            chooseThreads.addRecord(iSpecies, mStream);
            nSpeciesChosen[ iSpecies ] += 1;
            if ( diagMovesOn )
                this-> template launch_CheckSpecies<18>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
            else 
                this-> template launch_CheckSpecies<6>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
            	
            launch_CheckSpeciesInteraction(nBlocks, nThreads, iSpecies,seed );
            launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
            launch_ApplyInteraction(nBlocks, nThreads, iSpecies);
			
            if(met.getPacking().getNBufferedTmpLatticeOn()){
                /* we only need to delete when buffers will wrap around and
                    * on the last loop, so that on next runSimulationOnGPU
                    * call mLatticeTmp is clean */
                if ( ( iStepTotal % mnLatticeTmpBuffers == 0 ) ||
                    ( iStep == nMonteCarloSteps-1 && iSubStep == nSpecies-1 ) ){
                    cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
                }
            }else
                mLatticeTmp->memsetAsync(0);
            chooseThreads.analyze(iSpecies,mStream);
		} // iSubstep
    } // iStep
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
        << "run time (GPU): " << nMonteCarloSteps << "\n"
        << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
        << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    if (mLog.isActive( "Check" ) ){
        CUDA_ERROR( cudaStreamSynchronize( mStream ) );
        checkSystem(); // no-op if "Check"-level deactivated
        checkInteractionLatticeOccupation();
    }
    BaseClass::doCopyBack();
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template class UpdaterGPU_Interaction< uint8_t  >;
template class UpdaterGPU_Interaction< uint16_t >;
template class UpdaterGPU_Interaction< uint32_t >;
template class UpdaterGPU_Interaction<  int16_t >;
template class UpdaterGPU_Interaction<  int32_t >;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////