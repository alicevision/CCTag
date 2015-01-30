#ifndef _BOOST_GIL_COLOR_GRADATION_HPP_
#define _BOOST_GIL_COLOR_GRADATION_HPP_

#include <boost/gil/algorithm.hpp>
#include <terry/typedefs.hpp>
#include <terry/channel.hpp>

#include <cmath>

namespace terry {
using namespace boost::gil;

namespace color {

/**
 * @brief All supported gradations
 */
namespace gradation{
struct Linear{};
struct sRGB{};
struct Cineon
{
	Cineon()
		: _blackPoint( 95.0 )
		, _whitePoint( 685.0 )
		, _gammaSensito( 300.0 )
	{}
	Cineon( const double blackPoint, const double whitePoint, const double gammaSensito )
		: _blackPoint( blackPoint )
		, _whitePoint( whitePoint )
		, _gammaSensito( gammaSensito )
	{}
	double _blackPoint;
	double _whitePoint;
	double _gammaSensito;
};
struct Gamma
{
	Gamma()
	: _gamma( 1.0 )
	{}
	Gamma( const double gamma )
	: _gamma(gamma)
	{}
	double _gamma;
};
struct Panalog{};
struct REDLog{};
struct ViperLog{};
struct REDSpace{};
struct AlexaLogC{};
}

/// @brief change the color gradation
template< typename Channel,
          class IN,
          class OUT >
struct channel_color_gradation_t : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	// This class implementation must be used only:
	// * with IN != OUT
	BOOST_STATIC_ASSERT(( ! boost::is_same<IN, OUT>::value )); // Must use channel_color_gradation_t<Channel, INOUT> !
	// * IN and OUT must be other gradation mode than Linear
	//   For each gradation mode, you have to specialize: GradationMode -> Linear and Linear -> GradationMode
	BOOST_STATIC_ASSERT(( ! boost::is_same<IN, gradation::Linear>::value )); // The conversion IN to Linear is not implemented !
	BOOST_STATIC_ASSERT(( ! boost::is_same<OUT, gradation::Linear>::value )); // The conversion Linear to OUT is not implemented !

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		Channel inter;
		// IN -> Linear
		channel_color_gradation_t<Channel, IN, gradation::Linear>( _in, gradation::Linear() )( src, inter );
		// Linear -> OUT
		return channel_color_gradation_t<Channel, gradation::Linear, OUT>( gradation::Linear(), _out )( inter, dst );
	}
};

template< typename Channel,
          class INOUT >
struct channel_color_gradation_t<Channel, INOUT, INOUT> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	const INOUT& _in;
	const INOUT& _out;

	channel_color_gradation_t( const INOUT& in, const INOUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		return dst = Channel(src);
	}
};

////////////////////////////////////////////////////////////////////////////////
// sRGB //

/**
 * @brief sRGB to Lin
 * 
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::sRGB, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::sRGB IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst;

		if( src > 0.04045 )
		{
			fDst = std::pow( ( fSrc + 0.055 ) / 1.055 , 2.4 );
		}
		else
		{
			fDst = fSrc / 12.92;
		}
		/*
		// Old sRGB standard, taken from Computational color technology by Kang.
		if( fSrc > 0.03928 )
		{
			fDst = exp( log( ( fSrc + 0.055 ) / 1.055 ) * 2.4 );
		}
		else
		{
			fDst = fSrc / 12.92;
		}
		*/
		return dst = channel_convert<Channel>( fDst );
	}

};

/**
 * @brief Lin to sRGB
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::sRGB> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::sRGB OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst;

		if( src > 0.0031308 )
		{
			fDst = 1.055 * pow( fSrc , 1.0 / 2.4 ) - 0.055;
		}
		else
		{
			fDst = 12.92 * fSrc;
		}
		/*
		// Old sRGB standard, taken from Computational color technology by Kang.
		static const double inv_2_4 = 1.0 / 2.4;
		if( fSrc > 0.00304 )
		{
			fDst = 1.055 * exp( log( fSrc ) * inv_2_4 ) - 0.055;
		}
		else
		{
			fDst = 12.92 * fSrc;
		}
		*/
		return dst = channel_convert<Channel>( fDst );
	}

};


////////////////////////////////////////////////////////////////////////////////
// Cineon

/**
 * @brief Cineon to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Cineon, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Cineon IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );

		// double tmpValue = std::pow( 10 , ( _in._blackPoint - _in._whitepoint ) / 300 );
		// fDst = ( std::pow( 10 , ( fSrc*1024 - _in._whitePoint ) / 300.0 ) - tmpValue ) / tmpValue;
		T fDst = 1.010915615730753 * ( std::pow( 10.0 ,( 1023 * fSrc - 685 ) / 300.0 ) - 0.010797751623277 );

		return dst = channel_convert<Channel>( fDst );
	}
};

/**
 * @brief Lin to Cineon
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::Cineon> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::Cineon OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		// const double tmpValue = pow( 10.0, ( _out._blackPoint - _out._whitepoint ) / 300.0 );
		// fDst = ( std::pow( 10.0, ( fSrc*1024 - _out._whitePoint ) / 300.0 ) - tmpValue ) / tmpValue;
		T fDst = 0.00042453028534042214 * ( 300 * std::log( 0.98920224837672 * fSrc + 0.010797751623277 ) + 1577.270788700921 );

		return dst = channel_convert<Channel>( fDst );
	}
};

////////////////////////////////////////////////////////////////////////////////
// Gamma

/**
 * @brief Gamma to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Gamma, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Gamma IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst;

		if( src == 0 )
		{
			fDst = 0.0;
		}
		else if( src > 0.0 )
		{
			// compute gamma value
			// fDst = DstChannel( std::pow( 10.0, _value * std::log10( fSrc ) ) );
			fDst = std::pow( static_cast<double>(fSrc), _in._gamma );
		}
		else
		{
			// for negative values, we return a linear conversion
			fDst = fSrc;
		}
		return dst = channel_convert<Channel>( fDst );
	}
};

/**
 * @brief Lin to Gamma
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::Gamma> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::Gamma OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst;

		if( src == 0 )
		{
			fDst = 0.0;
		}
		else if( src > 0.0 )
		{
			// compute gamma value
			fDst = std::pow( static_cast<double>(fSrc) , 1.0 / _out._gamma );
		}
		else
		{
			// for negative values, we return a linear conversion
			fDst = fSrc;
		}
		return dst = channel_convert<Channel>( fDst );
	}
};


////////////////////////////////////////////////////////////////////////////////
// Panalog

/**
 * @brief Panalog to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Panalog, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Panalog IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst = ( std::pow ( 10.0, ( 1023.0 * fSrc - 681.0 ) / 444.0 ) - 0.0408 ) / ( 1.0 - 0.0408 );
		return dst = channel_convert<Channel>( fDst );
	}
};

/**
 * @brief Lin to Panalog
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::Panalog> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::Panalog OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst = ( 444.0 * std::log10( 0.0408 + ( 1.0 - 0.0408 ) * fSrc ) + 681.0 ) / 1023.0;
		return dst = channel_convert<Channel>( fDst );
	}
};


////////////////////////////////////////////////////////////////////////////////
// REDLog

/**
 * @brief REDLog to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::REDLog, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::REDLog IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst = ( std::pow ( 10.0, ( 1023.0 * fSrc - 1023.0 ) / 511.0 ) - 0.01 ) / ( 1.0 - 0.01 );
		return dst = channel_convert<Channel>( fDst );
	}
};

/**
 * @brief Lin to REDLog
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::REDLog> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::REDLog OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst = ( 511.0 * std::log10( 0.01 + ( 1.0 - 0.01 ) * fSrc ) + 1023.0 ) / 1023.0;
		return dst = channel_convert<Channel>( fDst );
	}
};


////////////////////////////////////////////////////////////////////////////////
// ViperLog

/**
 * @brief ViperLog to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::ViperLog, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::ViperLog IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst = std::pow( 10.0, ( 1023.0 * fSrc - 1023.0 ) / 500.0 );
		return dst = channel_convert<Channel>( fDst );
	}
};

/**
 * @brief Lin to ViperLog
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::ViperLog> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::ViperLog OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		const T fSrc = channel_convert<T>( src );
		T fDst = ( 500.0 * std::log10( fSrc ) + 1023.0 ) / 1023.0;
		return dst = channel_convert<Channel>( fDst );
	}
};


////////////////////////////////////////////////////////////////////////////////
// REDSpace

/**
 * @brief REDSpace to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::REDSpace, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::REDSpace IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		return dst = Channel( src ); // the equation wasn't found actually
	}
};

/**
 * @brief Lin to REDSpace
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::REDSpace> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::REDSpace OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		return dst = Channel( src ); // the equation wasn't found actually
	}
};


////////////////////////////////////////////////////////////////////////////////
// AlexaLogC

/**
 * @brief AlexaLogC to Lin
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::AlexaLogC, gradation::Linear> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::AlexaLogC IN;
	typedef typename gradation::Linear OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		return dst = Channel( src ); // the equation wasn't found actually
	}
};

/**
 * @brief Lin to AlexaLogC
 */
template< typename Channel >
struct channel_color_gradation_t<Channel, gradation::Linear, gradation::AlexaLogC> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_base_type<Channel>::type TBase;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;
	typedef typename gradation::Linear IN;
	typedef typename gradation::AlexaLogC OUT;

	const IN& _in;
	const OUT& _out;

	channel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		return dst = Channel( src ); // the equation wasn't found actually
	}
};



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


template< typename Pixel,
          class IN,
          class OUT >
struct pixel_color_gradation_t
{
	typedef typename channel_type<Pixel>::type Channel;
	const IN&  _in;
	const OUT& _out;
	
	pixel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}
	
	Pixel& operator()( const Pixel& p1,
	                   Pixel& p2 ) const
	{
		static_for_each(
				p1, p2,
				channel_color_gradation_t< Channel, IN, OUT >( _in, _out )
			);
		return p2;
	}
};

template< class IN,
          class OUT >
struct transform_pixel_color_gradation_t
{
	const IN&  _in;
	const OUT& _out;

	transform_pixel_color_gradation_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	template< typename Pixel>
	Pixel operator()( const Pixel& p1 ) const
	{
		Pixel p2;
		pixel_color_gradation_t<Pixel, IN, OUT>( _in, _out )( p1, p2 );
		return p2;
	}
};

/**
 * @example gradation_convert_view( srcView, dstView, gradation::sRGB(), gradation::Gamma(5.0) );
 */
template<class GradationIN, class GradationOUT, class View>
void gradation_convert_view( const View& src, View& dst, const GradationIN& gradationIn = GradationIN(), const GradationOUT& gradationOut = GradationOUT() )
{
	boost::gil::transform_pixels( src, dst, transform_pixel_color_gradation_t<GradationIN, GradationOUT>( gradationIn, gradationOut ) );
}

/**
 * @example gradation_convert_pixel( srcPix, dstPix, gradation::sRGB(), gradation::Gamma(5.0) );
 */
template<class GradationIN, class GradationOUT, class Pixel>
void gradation_convert_pixel( const Pixel& src, Pixel& dst, const GradationIN& gradationIn = GradationIN(), const GradationOUT& gradationOut = GradationOUT() )
{
	pixel_color_gradation_t<Pixel, GradationIN, GradationOUT>( gradationIn, gradationOut )( src, dst );
}


}
}

#endif
