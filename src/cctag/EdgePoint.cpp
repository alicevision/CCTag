#include "EdgePoint.hpp"

namespace cctag
{
namespace vision
{

std::ostream& operator<<( std::ostream& os, const EdgePoint& eP )
{
  os << "quiver( " << eP.x() << " , " << eP.y() << "," << eP._grad.x()/50 << "," << eP._grad.y()/50 << " ); ";
  return os;
}

}
}
