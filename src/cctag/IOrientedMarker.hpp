#ifndef _CCTAG_VISION_IORIENTEDMARKER_HPP_
#define _CCTAG_VISION_IORIENTEDMARKER_HPP_

#include "IMarker.hpp"

#include <cctag/algebra/matrix/Matrix.hpp>

#include <opencv2/core/types_c.h>

namespace rom {
namespace vision {
namespace marker {

/**
 * @brief Interface for oriented markers
 */
class IOrientedMarker : public IMarker
{
public:
	IOrientedMarker();
	IOrientedMarker( const rom::Point2dN<double> & centerImg )
        : IMarker( centerImg )
        {}
	IOrientedMarker( const IOrientedMarker & m )
	: IMarker( m )
	{ }
	virtual ~IOrientedMarker() = 0;

	virtual IOrientedMarker* clone() const = 0;

public:
        virtual inline const rom::numerical::BoundedMatrix3x3d & homography() const = 0;

};

inline IOrientedMarker* new_clone( const IOrientedMarker& x )
{
	return x.clone();
}

}
}
}

#endif
