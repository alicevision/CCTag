#include <cctag/EdgePoint.hpp>

namespace cctag
{

std::ostream& operator<<( std::ostream& os, const EdgePoint& eP )
{
  os << "quiver( " << eP.x() << " , " << eP.y() << "," << eP.dX() << "," << eP.dY() << " ); ";
  return os;
}

} // namespace cctag
