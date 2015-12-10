#pragma once

#include "cctag/CCTagMarkersBank.hpp"
#include "cctag/params.hpp"

#include <list>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

namespace popart
{
class Package;

class PackagePool
{
    const cctag::Parameters&       _params;
    const cctag::CCTagMarkersBank& _bank;
    int                            _max_packages;
    int                            _allocated_packages;
    std::list<Package*>            _free_packages;
    boost::mutex                   _lock;
    boost::condition               _cond;

public:
    PackagePool( const cctag::Parameters & params,
                 const cctag::CCTagMarkersBank & bank,
                 int   max_packages );

    Package* getPackage();
    void     returnPackage( Package* package );
    void     waitAllPackagesIdle( );
};

} //namespace popart
