#ifndef _TERRY_COLOR_GRADIENT_HPP_
#define _TERRY_COLOR_GRADIENT_HPP_

#include <terry/channel.hpp>

#include <boost/math/constants/constants.hpp>


namespace terry {
namespace color {

/**
 * @brief Compute the direction from the (x, y) coordinates of the input vector.
 */
template< typename Channel>
struct channel_gradientDirection_t
{
	GIL_FORCEINLINE
	void operator()( const Channel& x, const Channel& y, Channel& res ) const
	{
		res = std::atan2( (double)y, (double)x );
	}
};

/**
 * @brief Compute the direction from the (x, y) coordinates of the input vector, limited between 0 and PI.
 */
template< typename Channel>
struct channel_gradientDirectionAbs_t
{
	GIL_FORCEINLINE
	void operator()( const Channel& x, const Channel& y, Channel& res ) const
	{
		channel_gradientDirection_t<Channel>()(x, y, res);
		if( res < 0 )
			res += boost::math::constants::pi<typename channel_base_type<Channel>::type>();
	}
};

}
}

#endif
