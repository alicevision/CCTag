#include "cuda/cmp_list.h"

using namespace popart;

__host__
vote_index_sort::vote_index_sort( const HostEdgeList<TriplePoint>& voters )
    : _voters( voters )
{ }

std::ostream& operator<<( std::ostream& ostr, const int2& v )
{
    ostr << "(" << v.x << "," << v.y << ")";
    return ostr;
}

