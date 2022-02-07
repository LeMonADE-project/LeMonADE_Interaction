#ifndef COMMANDLINE_PARSER_H
#define COMMANDLINE_PARSER_H
#include<stdint.h>
#include<stdexcept>
#include<string>
#include<sstream>
#include<vector>
class CommandOptionObject
{
  friend class CommandLineParser;
public:
  CommandOptionObject();
  CommandOptionObject(std::string name,size_t nParameters,std::string helpText);
	
  virtual ~CommandOptionObject();
	
  std::string getName() const;
  std::string getHelpText() const;
  size_t getNParameters() const;
  //const char* operator[](size_t n) const;
  std::string operator[](size_t n) const;
	
private:
	
  bool isActive;
  std::string name;
  std::string helpText;
  size_t nParameters;
  //std::vector<const char*> parsedOption;
  std::vector<std::string> parsedOption;
};


class CommandLineParser
{
public:
  CommandLineParser();
  virtual ~CommandLineParser();
	
  void addOption(std::string name, size_t nParameters,std::string helpText);
  void parse(char* line[],size_t length);
  void displayHelp();
  std::vector<CommandOptionObject> activeOptions() const;
  //bool getOption(std::string name,std::vector<const char*>& returnParameters) const; 

  bool getOption(std::string name) const;
  template<typename ReturnType> 
    bool getOption(std::string name, ReturnType& retParam,size_t index=0) const;

    //a convenience function for getting filenames from input field
    std::vector<std::string> getInputFiles(char* line[],int32_t length) const;
private:
  std::vector<CommandOptionObject>::iterator findOption(std::string optName);
  std::vector<CommandOptionObject> registeredOptions;
	
};

template<typename ReturnType>
bool CommandLineParser::getOption(std::string name, ReturnType& retParam, size_t index) const
{
  for(size_t n=0;n<registeredOptions.size();n++)
    {
      if(registeredOptions[n].getName()==name && registeredOptions[n].isActive)
	{
	  if(index<registeredOptions[n].getNParameters()){
	    std::stringstream converter;
	    converter.exceptions(std::ios::failbit|std::ios::badbit);
	    converter<<registeredOptions[n].parsedOption[index];
	    try{
	      converter>>retParam;
	    }
	    catch(std::ios_base::failure& e ){
	      std::string origMessage(e.what());
	      std::stringstream errormessage;
	      errormessage<<"CommandLineParser::getOption(..): error converting "
			  <<registeredOptions[n].parsedOption[index]
			  <<" to required data type\n";
	      errormessage<<"(original error: "<<origMessage<<")";
	      throw  std::runtime_error(errormessage.str());
	      
	    }
	    return true;
	  }
	  else{
	    std::stringstream errormessage;
	    errormessage<<"CommandLineParser::getOption(string,ReturnType,size_t):"
			<<" index "<<index<<" out of range.\n";
	    throw std::runtime_error(errormessage.str());
	  }
	}
    }

  //if not found we are still here and can return false
  return false;
}


// a convenience function for getting filenames from input field

#endif /*COMMANDLINE_PARSER_H*/
