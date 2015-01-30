#ifndef _TERRY_COLOR_PRIMARIES_HPP_
#define	_TERRY_COLOR_PRIMARIES_HPP_

namespace terry {
namespace color {

/**
 * @brief All supported primaries
 */
namespace primaries {

}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if 0

template< typename Pixel,
          class IN,
          class OUT >
struct pixel_color_primaries_t
{
	typedef typename channel_type<Pixel>::type Channel;
	const IN&  _in;
	const OUT& _out;

	pixel_color_primaries_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	Pixel& operator()( const Pixel& p1,
	                   Pixel& p2 ) const
	{
		static_for_each(
				p1, p2,
				channel_color_primaries_t< Channel, IN, OUT >( _in, _out )
			);
		return p2;
	}
};

template< class IN,
          class OUT >
struct transform_pixel_color_primaries_t
{
	const IN&  _in;
	const OUT& _out;

	transform_pixel_color_primaries_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	template< typename Pixel>
	Pixel operator()( const Pixel& p1 ) const
	{
		Pixel p2;
		pixel_color_primaries_t<Pixel, IN, OUT>( _in, _out )( p1, p2 );
		return p2;
	}
};

/**
 * @example primaries_convert_view( srcView, dstView, primaries::sRGB(), primaries::Gamma(5.0) );
 */
template<class PrimariesIN, class PrimariesOUT, class View>
void primaries_convert_view( const View& src, View& dst, const PrimariesIN& primariesIn = PrimariesIN(), const PrimariesOUT& primariesOut = PrimariesOUT() )
{
	boost::gil::transform_pixels( src, dst, transform_pixel_color_primaries_t<PrimariesIN, PrimariesOUT>( primariesIn, primariesOut ) );
}

/**
 * @example primaries_convert_pixel( srcPix, dstPix, primaries::sRGB(), primaries::Gamma(5.0) );
 */
template<class PrimariesIN, class PrimariesOUT, class Pixel>
void primaries_convert_pixel( const Pixel& src, Pixel& dst, const PrimariesIN& primariesIn = PrimariesIN(), const PrimariesOUT& primariesOut = PrimariesOUT() )
{
	pixel_color_primaries_t<Pixel, PrimariesIN, PrimariesOUT>( primariesIn, primariesOut )( src, dst );
}

#endif

}
}

#endif

