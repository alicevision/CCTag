#ifndef _TERRY_GENERATOR_CHECKERBOARD_HPP_
#define _TERRY_GENERATOR_CHECKERBOARD_HPP_

#include <boost/gil/utilities.hpp>

#include <cmath>

namespace terry {
namespace generator {

// Models a Unary Function
template <typename P>
// Models PixelValueConcept
struct CheckerboardFunctor
{
	//typedef point2<ptrdiff_t>    point_t;
	typedef boost::gil::point2<double>    point_t;

	typedef CheckerboardFunctor const_t;
	typedef P value_type;
	typedef value_type reference;
	typedef value_type const_reference;
	typedef point_t argument_type;
	typedef reference result_type;
	BOOST_STATIC_CONSTANT( bool, is_mutable = false );

	value_type _in_color, _out_color;
	point_t _tile_size, _tile_size_2;

	CheckerboardFunctor() {}
	CheckerboardFunctor( const point_t& tileSize, const value_type& in_color, const value_type& out_color )
		: _in_color( in_color )
		, _out_color( out_color )
		, _tile_size( tileSize )
		, _tile_size_2( tileSize*2.0 ) {}

	result_type operator()( const point_t& p ) const
	{
		const point_t mp( fmod( p.x, _tile_size_2.x ), fmod( p.y, _tile_size_2.y ) );

		/*
		   std::cout << "__________" << std::endl;
		   std::cout << "p.x: " << p.x << " p.y: " << p.y << std::endl;
		   std::cout << "mp.x: " << mp.x << " mp.y: " << mp.y << std::endl;
		   std::cout << "_tile_size.x: " << _tile_size.x << " _tile_size.y: " << _tile_size.y << std::endl;
		 */
		if( ( mp.x > _tile_size.x ) != ( mp.y > _tile_size.y ) )
			return _in_color;
		return _out_color;
	}

};

}
}

#endif
