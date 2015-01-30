#ifndef _TERRY_COLOR_NORM_HPP_
#define _TERRY_COLOR_NORM_HPP_

#include <boost/math/special_functions/pow.hpp>

namespace terry {
namespace color {

/**
 * @brief Compute the norm from the (x, y) coordinates of the input vector.
 */
template<typename Channel>
struct channel_norm_t
{
	GIL_FORCEINLINE
	void operator()( const Channel& a, const Channel& b, Channel& res ) const
	{
		res = std::sqrt( boost::math::pow<2>(a) + boost::math::pow<2>(b) );
	}
};

/**
 * @brief Compute the Manhattan norm from the (x, y) coordinates of the input vector.
 */
template<typename Channel>
struct channel_normManhattan_t
{
	GIL_FORCEINLINE
	void operator()( const Channel& a, const Channel& b, Channel& res ) const
	{
		res = std::abs(a) + std::abs(b);
	}
};


}
}

#endif
