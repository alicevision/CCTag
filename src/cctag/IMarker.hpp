#ifndef _IMARKER_HPP
#define _IMARKER_HPP

#include <cctag/geometry/point.hpp>

namespace cctag {
namespace vision {
namespace marker {

typedef int MarkerID;
typedef std::vector< std::pair< MarkerID, double > > IdSet;

/**
 * @brief Interface for markers
 *
 */
class IMarker
//	: private boost::noncopyable
{
public:
	IMarker();
	IMarker( const cctag::Point2dN<double> & centerImg ): _centerImg( centerImg ) {}
	IMarker( const IMarker & m );
	virtual ~IMarker() = 0;

	bool operator==( const IMarker& m ) const
	{
		return this->_centerImg == m._centerImg &&
				this->id() == m.id();
	}

	/** @brief Set marker detection quality
	 * @param quality marker quality
	 */
	void setQuality( const int quality ) { _quality = quality; }

	/** @brief Set center position in original image
	 * @param center center position
	 */
	void setCenterImg( const cctag::Point2dN<double>& center ) { _centerImg = center; }

	/** @brief Set center position in the scene
	 * @param center center position
	 */
	void setCenterScene( const cctag::Point3d<double>& center ) { _centerScene = center; }

	/** @brief get marker detection quality
	 * @return quality reference
	 */
	int quality() const { return _quality; }

	/** @brief get center inside 2D image
	 * @return center 2D position
	 */
	const cctag::Point2dN<double>& centerImg() const { return _centerImg; }

	/** @brief get center inside 2D image
	 * @return center 2D position
	 */
	cctag::Point2dN<double>& centerImg() { return _centerImg; }

	/** @brief get center inside 3D scene
	 * @return center 3D position
	 */
	const cctag::Point3d<double>& centerScene() const { return _centerScene; }

	virtual bool hasId() const = 0;
	virtual MarkerID id() const = 0;
	virtual void setId( const MarkerID id ) = 0;

	virtual IMarker* clone() const = 0;

protected:
	cctag::Point2dN<double> _centerImg;     ///< Center of the marker inside original image
	cctag::Point3d<double> _centerScene;   ///< Center of the marker inside scene @todo REMOVE THIS !
	int _quality;                               ///< Marker quality
};


inline IMarker* new_clone( const IMarker& x )
{
	return x.clone();
}

}
}
}

#endif
