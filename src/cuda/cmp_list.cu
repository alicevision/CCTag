#include "cuda/cmp_list.h"

using namespace popart;

std::ostream& operator<<( std::ostream& ostr, const int2& v )
{
    ostr << "(" << v.x << "," << v.y << ")";
    return ostr;
}

std::ostream& operator<<( std::ostream& ostr, const short2& v )
{
    ostr << "(" << v.x << "," << v.y << ")";
    return ostr;
}

