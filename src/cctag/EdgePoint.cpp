#include "EdgePoint.hpp"

namespace popart {
namespace vision {

std::ostream& operator<<( std::ostream& os, const EdgePoint& eP )
{

	//Point2dN<int> xy eP.Point2dN();
	//Point2dN<double> grad = eP.gradient();

	os << "quiver( " << eP.x() << " , " << eP.y() << "," << eP._grad.x()/50 << "," << eP._grad.y()/50 << " ); ";

	return os;
}

}
}
