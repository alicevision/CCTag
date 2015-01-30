#ifndef _TERRY_BASICCOLORS_HPP_
#define	_TERRY_BASICCOLORS_HPP_

#include <boost/gil/pixel.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/color_convert.hpp>

namespace terry {

/**
 * @todo tuttle: to rewrite !!!
 */
struct alpha_max_filler
{
	template< class P>
	inline P operator()( const P& p ) const
	{
		using namespace boost::gil;
		gil_function_requires<ColorSpacesCompatibleConcept<
		                          typename color_space_type<P>::type,
		                          rgba_t> >( );
		P p2;
		p2[3] = channel_traits< typename channel_type< P >::type >::max_value();
		return p2;
	}

};

/**
 * @todo tuttle: to rewrite !!!
 */
struct black_filler
{
	template< class P>
	inline P operator()( const P& p ) const
	{
		using namespace boost::gil;
		P p2;
		for( int v = 0; v < num_channels<P>::type::value; ++v )
		{
			p2[v] = 0;
		}
		return p2;
	}

};

/**
 * @brief Get black color value
 */
template<class Pixel>
static inline const Pixel get_black()
{
	using namespace boost::gil;
	Pixel black;
	/// @todo tuttle: to rewrite !!!
	boost::gil::color_convert( gray32f_pixel_t( 0.0 ), black );
	return black;
}

template<class View>
static inline const typename View::value_type get_black( const View& )
{
	return get_black<typename View::value_type>();
}

template<class View>
static inline typename View::value_type get_white()
{
	using namespace boost::gil;
	typename View::value_type white;
	/// @todo tuttle: to rewrite !!!
	boost::gil::color_convert( gray32f_pixel_t( 1.0 ), white );
	return white;
}

template<class View>
static inline typename View::value_type get_white( const View& )
{
	return get_white<View>();
}

template <class View>
void fill_alpha_max( const View& v )
{
	using namespace boost::gil;
	transform_pixels( v, v, alpha_max_filler() );
}

/**
 * @brief Fill an image in black, all channels to 0.0 value and alpha channel to 1.0 (if exists)
 * @todo tuttle: to rewrite !!!
 */
template <class View>
void fill_black( View& v )
{
	using namespace boost::gil;
	transform_pixels( v, v, black_filler() );
	// Following doesn't work for built-in pixel types
	//	fill_pixels( v, get_black( v ) );
}


}

#endif

