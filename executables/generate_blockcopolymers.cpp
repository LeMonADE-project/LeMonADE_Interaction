#include <LeMonADE/core/Ingredients.h>
#include <LeMonADE/feature/FeatureMoleculesIOUnsaveCheck.h>
#include <LeMonADE/feature/FeatureExcludedVolumeSc.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/analyzer/AnalyzerWriteBfmFile.h>
#include <LeMonADE/utility/RandomNumberGenerators.h>
#include <LeMonADE/utility/TaskManager.h>
#include <LeMonADE/feature/FeatureNNInteractionSc.h>
#include <LeMonADE/analyzer/AnalyzerWriteBfmFileSubGroup.h>
#include <LeMonADE/utility/DepthIteratorPredicates.h>
#include <LeMonADE/feature/FeatureSystemInformationLinearMeltWithCrosslinker.h>


#include <LeMonADE_Interaction/updater/UpdaterAddLinearChains.h>
#include <LeMonADE_Interaction/updater/UpdaterAddInteractionTagLC.h>
#include <LeMonADE_Interaction/utility/CommandlineParser.h>

/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    std::string input("config.bfm");
    std::string output("output.bfm");
    uint32_t nChains, chainLength, nSolventMolecules;
    double energyAS(0.), energyBS(0.), energyAB(0.);
    uint32_t tag1(2),tag2(3),box(128);
    bool periodic(false);
    try 
    {
        //////////////////////////////////////////////////////////////////////
        //parse command line arguments
        CommandLineParser cmd;
        
        //add possible options for command line
        cmd.addOption("-i" ,1,"input "                          );
        cmd.addOption("-o" ,1,"output"                          );

        cmd.addOption("-as",1,"energy AS (species A to solvent)");
        cmd.addOption("-bs",1,"energy BS (species B to solvent)");
        cmd.addOption("-ab",1,"energy AS (species A to B)"      );

        cmd.addOption("-s" ,1,"numer of solvent monomers"       );
        cmd.addOption("-n" ,1,"number of monomers per chain"    );
        cmd.addOption("-m" ,1,"number of chains "               );

        cmd.addOption("-p" ,0,"if given use periodic boundaries");
        cmd.addOption("-b" ,1,"box size "                       );

        cmd.addOption("-h" ,0,"display help"                    );
        
        //parse command line options
        cmd.parse(argv+1,argc-1);
        
        if(argc==1 || cmd.getOption("-h")){
                std::cout<<"****** generate_blockcopolymers  *************\n\n"
                <<"Usage: generate_blockcopolymers  [options]\n";
                cmd.displayHelp();
                exit(0);
        }
        
        cmd.getOption("-i", input   );
        cmd.getOption("-o", output  );

        cmd.getOption("-s", nSolventMolecules);
        cmd.getOption("-n", chainLength    );
        cmd.getOption("-m", nChains    );

        cmd.getOption("-ab", energyAB);
        cmd.getOption("-as", energyAS);
        cmd.getOption("-bs", energyBS);

        cmd.getOption("-b" ,box);
        periodic=cmd.getOption("-p");
        //////////////////////////////////////////////////////////////////////
    }
    catch(std::exception& e){std::cerr<<"Error:\n" <<e.what()<<std::endl;}
    catch(...){std::cerr<<"Error: unknown exception\n";}
    typedef LOKI_TYPELIST_4(
        FeatureMoleculesIOUnsaveCheck, 
        FeatureNNInteractionSc,
        FeatureExcludedVolumeSc< FeatureLatticePowerOfTwo <bool > >,
        FeatureSystemInformationLinearMeltWithCrosslinker
        ) Features;
    const uint max_bonds=2;
    typedef ConfigureSystem<VectorInt3,Features,max_bonds> Config;
    typedef Ingredients<Config> IngredientsType;
    IngredientsType ingredients;

    RandomNumberGenerators rng;
    rng.seedAll();
    ingredients.setBoxX(box);
    ingredients.setBoxY(box);
    ingredients.setBoxZ(box);
    ingredients.setPeriodicX(periodic);
    ingredients.setPeriodicY(periodic);
    ingredients.setPeriodicZ(periodic);
    ingredients.modifyBondset().addBFMclassicBondset();
    ingredients.setNumOfChains(nChains);
    ingredients.setNumOfMonomersPerChain(chainLength);
    ingredients.synchronize();

    TaskManager taskManager;
    
    //Read in the last config of the bfm file by iterating ('save') through all configs up to the last one
    // taskManager.addUpdater(new UpdaterReadBfmFile<IngredientsType>(input, ingredients, UpdaterReadBfmFile<IngredientsType>::READ_LAST_CONFIG_SAVE)); 
    taskManager.addUpdater(new UpdaterAddLinearChains<IngredientsType>(ingredients, nChains,chainLength));
    if (nSolventMolecules > 0)
        taskManager.addUpdater(new UpdaterAddLinearChains<IngredientsType>(ingredients, nSolventMolecules,1));
    taskManager.addUpdater(new UpdaterAddInteractionTag<IngredientsType>(ingredients, nChains,chainLength,tag1,tag2,energyAB,energyAS,energyBS));
        
    //append all snapshots to one file
    taskManager.addAnalyzer(new AnalyzerWriteBfmFile<IngredientsType>(output, ingredients, AnalyzerWriteBfmFile<IngredientsType>::APPEND));
    
    //run program
    taskManager.initialize();
    taskManager.run(1);
    taskManager.cleanup();
    
    return 0;

    /////////////////////////////////////////////////////////////////////////////////////

}

