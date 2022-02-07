/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo---0---oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADE.

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

#ifndef LEMONADEINTERACTION_UPDATER_ADD_INTERACTION_TAG_LC
#define LEMONADEINTERACTION_UPDATER_ADD_INTERACTION_TAG_LC
/**
 * @file
 *
 * @class UpdaterAddInteractionTag
 *
 * @brief Updater to add an inteaction tag to linear chains.
 *
 * @tparam IngredientsType
 *
 * @param ingredients_ The system, holding all monomers
 * @param nChain_ number of chains that are added to ingredients
 * @param nMonoPerChain_ number of monomer is each chain
  **/

#include <LeMonADE/updater/AbstractUpdater.h>
#include <LeMonADE/utility/Vector3D.h>
#include <cmath>

template<class IngredientsType>
class UpdaterAddInteractionTag: public AbstractUpdater
{

public:
    UpdaterAddInteractionTag(IngredientsType& ingredients_, uint32_t nChain_, uint32_t nMonoPerChain_, uint32_t tag1_, uint32_t tag2_, double energyAB_, double energyAS_, double energyBS_, uint32_t tagSolvent_=1 );

    virtual void initialize();
    virtual bool execute();
    virtual void cleanup(){};

private:

    //! container holding ingredients
    IngredientsType& ingredients;

    //! number of monomers in a chain
    uint32_t nMonoPerChain;

    //! number of linear chains in the box
    uint32_t nChain;
    
    //! bool for execution
    bool wasExecuted;

    //! interaction tag for monomer species A 
    uint32_t tag1;

    //! interaction tag for monomer species B
    uint32_t tag2;

    //! interaction tag for the solvent monomers
    uint32_t tagSolvent;

    //! interaction energye AB 
    double energyAB;
    //! interaction energye AS
    double energyAS;
    //! interaction energye BS 
    double energyBS;
};

/**
* @brief Constructor handling the new systems paramters
*
* @param ingredients_ a reference to the IngredientsType - mainly the system
* @param nChain_ number of chains to be added in the system instead of solvent
* @param nMonoPerChain_ number of monomers in one chain
*/
template < class IngredientsType >
UpdaterAddInteractionTag<IngredientsType>::UpdaterAddInteractionTag(
    IngredientsType& ingredients_, 
    uint32_t nChain_, 
    uint32_t nMonoPerChain_, 
    uint32_t tag1_, 
    uint32_t tag2_, 
    double energyAB_, 
    double energyAS_, 
    double energyBS_, 
    uint32_t tagSolvent_ ):
    ingredients(ingredients_), nChain(nChain_), nMonoPerChain(nMonoPerChain_), 
    energyAB(energyAB_), energyAS(energyAS_), energyBS(energyBS_),
    tag1(tag1_), tag2(tag2_),tagSolvent(tagSolvent_),  wasExecuted(false){}

/**
* @brief initialise function, calculate the target density to compare with at the end.
*
* @tparam IngredientsType Features used in the system. See Ingredients.
*/
template < class IngredientsType >
void UpdaterAddInteractionTag<IngredientsType>::initialize(){
  std::cout << "initialize UpdaterAddInteractionTag" << std::endl;
  execute();

}

/**
* @brief Execution of the system creation
*
* @tparam IngredientsType Features used in the system. See Ingredients.
*/
template < class IngredientsType >
bool UpdaterAddInteractionTag<IngredientsType>::execute(){
    if(wasExecuted)
        return true;
    ingredients.setNNInteraction(tag1,tag2,energyAB);
    ingredients.setNNInteraction(tag1,tagSolvent,energyAS);
    ingredients.setNNInteraction(tag2,tagSolvent,energyBS);
    std::cout << "execute UpdaterAddInteractionTag" << std::endl;

    //loop over chains and chain monomers and build it up
    for(uint32_t i=0;i<(nChain);i++){
        for(uint32_t j=0;j<(nMonoPerChain);j++){
            uint32_t ID(i*nMonoPerChain+j);
            if(j <nMonoPerChain/2 ) 
                ingredients.modifyMolecules()[ ID ].setInteractionTag(tag1);
            else 
                ingredients.modifyMolecules()[ ID ].setInteractionTag(tag2);
        }
    }

    for (auto i =0; i < 20 ; i ++)
        std::cout << "ingredients.modifyMolecules()[ "<< i << " ].getInteractionTag()=" << static_cast<uint32_t>(ingredients.modifyMolecules()[ i ].getInteractionTag()) << std::endl;
    

    ingredients.synchronize();
    wasExecuted=true;
    return true;
}
#endif /* LEMONADEINTERACTION_UPDATER_ADD_INTERACTION_TAG_LC */