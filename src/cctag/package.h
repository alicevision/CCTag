#pragma once

#include <iostream>
#include <string>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/ptr_container/ptr_list.hpp>

#include "cctag/fileDebug.hpp"
#include "cctag/visualDebug.hpp"
#include "cctag/progBase/exceptions.hpp"
#include "cctag/detection.hpp"
#include "cctag/view.hpp"
#include "cctag/image.hpp"
#include "cctag/types.hpp"
#include "cctag/EdgePoint.hpp"
#include "cctag/CCTag.hpp"
// #include "cuda/ptrstep.h"
// #include "cctag/cmdline.hpp"

namespace popart
{
class FramePackage;
class PackagePool;

class Package
{
private:
    Package( );
    Package( const Package& );
    Package& operator=( const Package& );

protected:
    Package( PackagePool* pool,
             const cctag::Parameters & params,
             const cctag::CCTagMarkersBank & bank );

    friend class PackagePool;

public:
    void lockPhase1( );
    void unlockPhase1( );
    void lockPhase2( ) { }
    void unlockPhase2( ) { }

    void resetTables( );

    void init( int            frameId,
               const cv::Mat* src,
               std::ostream*  output,
               std::string    debugFileName = "" );

    FramePackage* getFramePackage( int i ) {
        if( i >= _numLayers ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                    << "    trying to access more than " << _numLayers << " allocations" << std::endl;
            exit( -1 );
        }
        if( not _framePackage ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                    << "    FramePackages not allocated yet." << std::endl;
            exit( -1 );
        }
        return _framePackage[i];
    }

    void detect( ); // indicate thread that a detection loop can start
    void detectionThread( ); // thread entry point
    void detectionRun( ); // one detection run

private:
    PackagePool*                   _my_pool;
    int                            _frameId;
    const cv::Mat*                 _src;
    const cctag::Parameters&       _params;
    const cctag::CCTagMarkersBank& _bank;
    std::ostream*                  _output;
    std::string                    _debugFileName;
    int                            _numLayers;
    FramePackage**                 _framePackage;
    boost::ptr_list<cctag::CCTag>  _markers;
    bool                           _readyForDetection;
    boost::mutex                   _rfd_lock;
    boost::condition               _rfd_cond;
    boost::thread                  _detect_thread;

    /* This is not a joke.
     * We need a global lock for the CUDA resources because
     * we require all memory.
     */
    static boost::mutex            _lock_phase_1;

public:
    // temp storage for cctagMultiresDetection_inner
    cctag::WinnerMap*               winners;
    std::vector<cctag::EdgePoint*>* seeds;
};

} //namespace popart
