/// \file SObservable.cpp
/// \brief SObservable class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SObservable.h"

SObservable::SObservable(){

}

SObservable::~SObservable(){

}

void SObservable::addObserver(SObserver* observer){
    m_observers.push_back(observer);
}

void SObservable::notifyProgress(int value){
    for (unsigned int i = 0 ; i < m_observers.size() ; i++){
        m_observers[i]->progress(value);
    }
}

void SObservable::notify(std::string message, int type){
    for (unsigned int i = 0 ; i < m_observers.size() ; i++){
        m_observers[i]->message(message, type);
    }
}
