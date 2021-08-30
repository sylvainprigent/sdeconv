/// \file SProcessObserver.h
/// \brief SProcessObserver class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>

/// \class SObserver
/// \brief Define a common interface to observe a SProcess
class SObserver{

public:
    static int MessageTypeDefault; // default console
    static int MessageTypeHighlight; // green
    static int MessageTypeHeader; // bold
    static int MessageTypeWarning; // orange or yellow
    static int MessageTypeError; // red bold

public:
    SObserver();
    virtual ~SObserver();

public:
    /// \brief Notify progress
    /// \param[in] value Progress value in [0, 100]
    virtual void progress(int value) = 0;

    /// \brief Notify a message
    /// \param[in] message Message to notify
    /// \param[in] type Type of message (see SObserver::MessageTypeXXX for possible values)
    virtual void message(std::string message, int type =  SObserver::MessageTypeDefault) = 0;

};
