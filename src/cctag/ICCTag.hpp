#ifndef PONCTUALCCTAG_HPP
#define	PONCTUALCCTAG_HPP

#include <cctag/params.hpp>
#include <cctag/CCTagMarkersBank.hpp>

#include <boost/ptr_container/ptr_list.hpp>
#include <boost/gil/gil_all.hpp>

#include <opencv2/core/core.hpp>

namespace cctag {

namespace logtime {
struct Mgmt;
}
  
typedef int MarkerID;

class ICCTag
{
public:

	ICCTag()
		: _x( 0.0 )
		, _y( 0.0 )
		, _id( -1 )
	{ }
                
        virtual double x() const = 0;
        virtual double y() const = 0;
        virtual MarkerID id() const = 0;
        virtual int getStatus() const = 0;

	virtual ~ICCTag() {}

protected:
	double _x;
	double _y;
	MarkerID _id;
        int _status;
};

// implemeted in cctag/ICCTag.cpp
void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const cv::Mat & graySrc,
      logtime::Mgmt* durations = 0,
      const std::size_t nCrowns = 3,
      const std::string & parameterFile = "",
      const std::string & cctagBankFilename = "");

// implemeted in cctag/ICCTag.cpp
void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const cv::Mat & graySrc,
      const cctag::Parameters & params,
      logtime::Mgmt* durations = 0,
      const CCTagMarkersBank * bank = NULL);

void initCCTagDetection( );
}

#endif	/* PONCTUALCCTAG_HPP */

