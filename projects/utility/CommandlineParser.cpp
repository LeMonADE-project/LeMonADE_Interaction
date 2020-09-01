#include<stdexcept>
#include<iostream>
#include<sstream>
#include<cstring>
#include"CommandlineParser.h"

CommandOptionObject::CommandOptionObject()
  :isActive(false),name(""),helpText(""),nParameters(0)
{}

CommandOptionObject::CommandOptionObject(std::string optionName,size_t nParams,std::string help)
  :isActive(false),name(optionName),helpText(help),nParameters(nParams)
{}

CommandOptionObject::~CommandOptionObject()
{}

std::string CommandOptionObject::getName() const
{
  return name;
}

std::string CommandOptionObject::getHelpText() const
{
  return helpText;
}

size_t CommandOptionObject::getNParameters() const
{
  return nParameters;
}

std::string CommandOptionObject::operator[](size_t n) const
{
  if(isActive && nParameters!=0)
    return parsedOption[n];
  else 
    {
      std::stringstream errormessage;
      errormessage<<"CommandOptionObject::operator[]:"
		  <<"error accessing argument "<<n<<" of option "<<name<<std::endl;
      if(!isActive)
	errormessage<<"option is not active\n";
      else
	errormessage<<"index "<<n<<" is larger than number of accepted arguments "<<nParameters<<std::endl;

      throw std::runtime_error(errormessage.str());
	
    }
}



CommandLineParser::CommandLineParser()
{}

CommandLineParser::~CommandLineParser()
{}

void CommandLineParser::addOption(std::string name, size_t nParameters,std::string helpText)
{
  registeredOptions.push_back(CommandOptionObject(name,nParameters,helpText));
}

void CommandLineParser::parse(char* line[],size_t length)
{
  
  for(size_t n=0;n<length;n++)
    {
      
      std::string opt(line[n]);
      
      std::vector<CommandOptionObject>::iterator it=findOption(opt);
      
      if(it!=registeredOptions.end())
	{
	  if(n<length-it->getNParameters())
	    {
	      for(size_t l=1;l<=it->getNParameters();l++)
		{
		  it->parsedOption.push_back(std::string(line[n+l]));
		}
	      it->isActive=true;
	      n+=it->getNParameters();
	    }
	  else
	    {
	      std::stringstream errormessage;
	      errormessage<<"CommandLineParser::parse(..): option "<<line[n]<<" given too few arguments"<<std::endl;
	      throw std::runtime_error(errormessage.str());
	    } 
	  
	}
      else
	{
	  std::stringstream errormessage;
	  errormessage<<"CommandLineParser::parse(..): invalid option "<<line[n]<<std::endl;
	  throw std::runtime_error(errormessage.str());
	}
     
    }
}

void CommandLineParser::displayHelp()
{
  std::cout<<"\noptions are:\n";
  for(size_t n=0;n<registeredOptions.size();n++)
    {
      std::cout<<registeredOptions[n].getName()<<":\t"<<registeredOptions[n].getHelpText()<<std::endl;
    }
  
}

std::vector<CommandOptionObject> CommandLineParser::activeOptions() const
{
  std::vector<CommandOptionObject> options;
  for(size_t n=0;n<registeredOptions.size();n++)
    {
      if(registeredOptions[n].isActive)
	{
	  options.push_back(registeredOptions[n]);
	}
    }
  return options;
}

// bool CommandLineParser::getOption(std::string name, std::vector<const char*>& returnParameters) const
// {
//   for(size_t n=0;n<registeredOptions.size();n++)
//     {
//       if(registeredOptions[n].getName()==name && registeredOptions[n].isActive)
// 	{
// 	  returnParameters=registeredOptions[n].parsedOption;
// 	  return true;
// 	}
//     }

//   //if not found we are still here and can return false
//   returnParameters.clear();
//   return false;
// }



bool CommandLineParser::getOption(std::string name) const
{
  for(size_t n=0;n<registeredOptions.size();n++)
    {
      if(registeredOptions[n].getName()==name && registeredOptions[n].isActive)
	{
	  return true;
	}
    }
  //if not found we are still here and can return false
  return false;
}


std::vector<CommandOptionObject>::iterator CommandLineParser::findOption(std::string optName)
{
  std::vector<CommandOptionObject>::iterator it;
  for(it=registeredOptions.begin();it!=registeredOptions.end();++it)
    {
      if(it->getName()==optName) return it;
    }

  return it;
}

std::vector<std::string> CommandLineParser::getInputFiles(char* line[],int32_t length) const
{
	std::vector<std::string> filenames;
	
	//get filenames
	size_t nFiles;
	for(nFiles=1;nFiles<length;nFiles++){
		if(std::strncmp(line[nFiles],"--",2)!=0)
			filenames.push_back(std::string(line[nFiles]));
		
		else break;
	}
	return filenames;
	
}

