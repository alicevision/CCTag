#ifndef _TERRY_ALGORITHM_FOREACH_HPP_
#define	_TERRY_ALGORITHM_FOREACH_HPP_

#include <terry/math/Rect.hpp>
#include <boost/gil/algorithm.hpp>

namespace terry {
namespace algorithm {

/// \brief std::for_each for a pair of iterators
template <typename Iterator1, typename Iterator2, typename BinaryFunction>
GIL_FORCEINLINE
BinaryFunction for_each( Iterator1 first1, Iterator1 last1, Iterator2 first2, BinaryFunction f )
{
    while( first1 != last1 )
	{
        f( *first1++, *first2++ );
	}
    return f;
}

}
}

#endif

