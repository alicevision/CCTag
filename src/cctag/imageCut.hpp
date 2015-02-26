#ifndef _CCTAG_LINE_HPP_
#define	_CCTAG_LINE_HPP_

#include <cctag/geometry/point.hpp>

#include <boost/numeric/ublas/vector.hpp>

namespace rom {

struct ImageCut
{
	Point2dN<double> _start;
	Point2dN<double> _stop;
	boost::numeric::ublas::vector<double> _imgSignal; //< image signal
};

}


#endif
