#include "cctag/packagePool.h"
#include "cctag/package.h"

namespace popart
{

PackagePool::PackagePool( const cctag::Parameters & params,
                          const cctag::CCTagMarkersBank & bank,
                          int max_packages )
    : _params( params )
    , _bank( bank )
    , _max_packages( max_packages )
    , _allocated_packages( 0 )
{
}

Package* PackagePool::getPackage( )
{
    Package* p;

    _lock.lock();
    if( not _free_packages.empty() ) {
        p = _free_packages.front();
        _free_packages.pop_front();
        _lock.unlock();
        return p;
    } else if( _allocated_packages < _max_packages ) {
        _lock.unlock();
        p = new Package( this, _params, _bank );
        _allocated_packages += 1;
        return p;
    } else {
        while( _free_packages.empty() ) {
            _cond.wait( _lock );
        }
        p = _free_packages.front();
        _free_packages.pop_front();
        _lock.unlock();
        return p;
    }
}

void PackagePool::returnPackage( Package* package )
{
    _lock.lock();
    _free_packages.push_back( package );
    _cond.notify_all();
    _lock.unlock();
}

void PackagePool::waitAllPackagesIdle( )
{
    _lock.lock();
    while( _free_packages.size() < _allocated_packages ) {
        _cond.wait( _lock );
    }
    _lock.unlock();
}

} //namespace popart
