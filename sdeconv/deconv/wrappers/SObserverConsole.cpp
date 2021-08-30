/// \file SObserverConsole.cpp
/// \brief SObserverConsole class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SObserverConsole.h"
#include <iostream>

SObserverConsole::SObserverConsole(){
    m_inProgress = false;
}

SObserverConsole::~SObserverConsole(){

}

void SObserverConsole::progress(int value){

    int barWidth = 70;
    std::cout << "";
    int pos = barWidth * value/100;
    std::cout << "\e[7m";
    for (int i = 0 ; i <= pos ; i++ ){
        if ( i < barWidth ){
            std::cout << " ";
        }
    }
    std::cout << "\e[27m";
    for (int i = pos+1 ; i < barWidth ; i++ ){
        std::cout << " ";
    }

    std::cout << "| " << value << " %\r";
    std::cout.flush();

    if (value == 100){
        std::cout << std::endl;
        m_inProgress = false;
    }
    else{
        m_inProgress = true;
    }
}

void SObserverConsole::message(std::string message, int type){

    if (m_inProgress){
        std::cout << std::endl;
        m_inProgress = false;
    }

    if (type == SObserver::MessageTypeDefault){
        std::cout << message << std::endl;
    }
    else if (type == SObserver::MessageTypeHighlight){
        std::cout << "\e[1m\e[92m" << message << "\e[0m" << std::endl;
    }
    else if (type == SObserver::MessageTypeHeader){
        std::cout << "\e[1m" << message << "\e[0m" << std::endl;
    }
    else if (type == SObserver::MessageTypeWarning){
        std::cout << "\e[1m\e[93m" << message << "\e[0m" << std::endl;
    }
    else if (type == SObserver::MessageTypeError){
        std::cout << "\e[1m\e[91m" << "ERROR: " <<  message << "\e[0m" << std::endl;
    }
    else{
        std::cout << message << std::endl;
    }


}
