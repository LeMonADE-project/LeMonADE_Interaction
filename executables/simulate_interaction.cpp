#include <LeMonADE/core/Ingredients.h>
#include <LeMonADE/feature/FeatureMoleculesIOUnsaveCheck.h>
#include <LeMonADE/feature/FeatureAttributes.h>
#include <LeMonADE/feature/FeatureExcludedVolumeSc.h>
#include <LeMonADE/updater/UpdaterSimpleSimulator.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/analyzer/AnalyzerWriteBfmFile.h>
#include <LeMonADE/utility/RandomNumberGenerators.h>
#include <LeMonADE/utility/TaskManager.h>
// #include <LeMonADE/utility/DepthIterator.h>
#include <LeMonADE/feature/FeatureNNInteractionSc.h>
#include <LeMonADE/analyzer/AnalyzerWriteBfmFileSubGroup.h>
#include <LeMonADE/utility/DepthIteratorPredicates.h>

#include <LeMonADE_Interaction/utility/CommandlineParser.h>
/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  int nMCS(100),interval(1), eqTime(0);
  std::string input("config.bfm");
  std::string output("output.bfm");
  
  try 
  {
        //////////////////////////////////////////////////////////////////////
        //parse command line arguments
        CommandLineParser cmd;
        
        //add possible options for command line
        cmd.addOption("-i" ,1,"input "             );
        cmd.addOption("-o" ,1,"output"             );
        cmd.addOption("-s" ,1,"interval/step size" );
        cmd.addOption("-m" ,1,"maximum MCS"        );
        cmd.addOption("-h" ,0,"display help" );
        
        
        //parse command line options
        cmd.parse(argv+1,argc-1);
        
        if(argc==1 || cmd.getOption("-h")){
                std::cout<<"****** simulateInteraction  *************\n\n"
                <<"Usage: simulateInteraction  [options]\n";
                cmd.displayHelp();
                exit(0);
        }
        
        cmd.getOption("-i", input   );
        cmd.getOption("-o", output  );
        cmd.getOption("-s", interval);
        cmd.getOption("-m", nMCS    );
        //////////////////////////////////////////////////////////////////////
    }
    catch(std::exception& e){std::cerr<<"Error:\n" <<e.what()<<std::endl;}
    catch(...){std::cerr<<"Error: unknown exception\n";}
    typedef LOKI_TYPELIST_3(
        FeatureMoleculesIOUnsaveCheck, 
        FeatureNNInteractionSc,
        FeatureExcludedVolumeSc< FeatureLatticePowerOfTwo <bool > >
        ) Features;
    const uint max_bonds=4;
    typedef ConfigureSystem<VectorInt3,Features,max_bonds> Config;
    typedef Ingredients<Config> IngredientsType;
    IngredientsType ingredients;

    RandomNumberGenerators rng;
    rng.seedAll();

    TaskManager taskManager;
    
    //Read in the last config of the bfm file by iterating ('save') through all configs up to the last one
    taskManager.addUpdater(new UpdaterReadBfmFile<IngredientsType>(input,ingredients,UpdaterReadBfmFile<IngredientsType>::READ_LAST_CONFIG_SAVE)); 
    
    // simulate the system 
    taskManager.addUpdater(new UpdaterSimpleSimulator<IngredientsType,MoveLocalSc>(ingredients,interval));
        
    //append all snapshots to one file
    taskManager.addAnalyzer(new AnalyzerWriteBfmFile<IngredientsType>(output          ,ingredients,AnalyzerWriteBfmFile<IngredientsType>::APPEND));
    
    //write out last config to a single file
    std::string lastConfigFileName=(output.substr(0,output.find_last_of("."))+"_lastconfig.bfm");
    taskManager.addAnalyzer(new AnalyzerWriteBfmFile<IngredientsType>(lastConfigFileName,ingredients,AnalyzerWriteBfmFile<IngredientsType>::OVERWRITE ));

    //Write out config without solvent
    std::string noSolventFileName=(output.substr(0,output.find_last_of("."))+"_nosolv.bfm");
    taskManager.addAnalyzer(new AnalyzerWriteBfmFileSubGroup<IngredientsType, hasBonds>(noSolventFileName, ingredients));
    
    //Run all subprograms nMCS/interval  number of times.
    taskManager.initialize();
    taskManager.run(nMCS/interval);
    taskManager.cleanup();
    
    return 0;

    /////////////////////////////////////////////////////////////////////////////////////

}

