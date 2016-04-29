#include "Operation.hpp"

#include <Eigen/LU>

namespace cctag {
namespace numerical {

Eigen::Matrix3f& normalizeDet1( Eigen::Matrix3f& m )
{

	const float det = m.determinant();
	if( det == 0 )
		return m;

	const float s = ( ( det >= 0 ) ? 1 : -1 ) / std::pow( std::abs( det ), 1.f / 3.f );
	m = s * m;
	return m;
}

} // numerical
} // cctag