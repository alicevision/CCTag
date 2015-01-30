#ifndef _TERRY_COLOR_TEMPERATURE_HPP_
#define	_TERRY_COLOR_TEMPERATURE_HPP_

#include <terry/copy.hpp>

namespace terry {
namespace color {

/**
 * @bief All supported temperatures
 */
namespace temperature {
struct T_A {};
struct T_B {};
struct T_C {};
struct T_D50 {};
struct T_D55 {};
struct T_D58 {};
struct T_D65 {};
struct T_D75 {};
struct T_9300 {};
struct T_E {};
struct T_F2 {};
struct T_F7 {};
struct T_F11 {};
struct T_DCIP3 {};

// we use D65 as the intermediate temperature color
typedef T_D65 T_INTER;
}


/// @brief change the color temperature
template< typename Channel,
          class IN,
          class OUT >
struct channel_color_temperature_t : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	// This class implementation must be used only:
	// * with IN != OUT
	BOOST_STATIC_ASSERT(( ! boost::is_same<IN, OUT>::value )); // Must use channel_color_temperature_t<Channel, INOUT> !
	// * IN and OUT must be other temperature mode than T_INTER
	//   For each temperature mode, you have to specialize: TemperatureMode -> T_INTER and T_INTER -> TemperatureMode
	BOOST_STATIC_ASSERT(( ! boost::is_same<IN, temperature::T_INTER>::value )); // The conversion IN to T_INTER is not implemented !
	BOOST_STATIC_ASSERT(( ! boost::is_same<OUT, temperature::T_INTER>::value )); // The conversion T_INTER to OUT is not implemented !

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		Channel inter;
		// IN -> T_INTER
		channel_color_temperature_t<Channel, IN, temperature::T_INTER>( _in, temperature::T_INTER() )( src, inter );
		// T_INTER -> OUT
		return channel_color_temperature_t<Channel, temperature::T_INTER, OUT>( temperature::T_INTER(), _out )( inter, dst );
	}
};

template< typename Channel,
          class INOUT >
struct channel_color_temperature_t<Channel, INOUT, INOUT> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	const INOUT& _in;
	const INOUT& _out;

	channel_color_temperature_t( const INOUT& in, const INOUT& out )
	: _in(in)
	, _out(out)
	{}

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		return dst = Channel(src);
	}
};

////////////////////////////////////////////////////////////////////////////////
// T_A //

/**
 * @brief 
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_A, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.84520578f * get_color( src, red_t() ) + 0.00000009f * get_color( src, green_t() ) + 0.00000002 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000005f * get_color( src, red_t() ) + 0.82607192f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000001f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.23326041 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief 
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_A> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.54194504f * get_color( src, red_t() ) + 0.00000004f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= +0.00000008f * get_color( src, red_t() ) + 1.21054840f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	= -0.00000001f * get_color( src, red_t() ) - 0.00000006f * get_color( src, green_t() ) + 0.85470724 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_B //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_B, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.24874067f * get_color( src, red_t() ) + 0.00000004f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 0.95096886f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.75399793 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_B> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.80080682f * get_color( src, red_t() ) + 0.00000004f * get_color( src, green_t() ) - 0.00000006 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000008f * get_color( src, red_t() ) + 1.05155921f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000001f * get_color( src, red_t() ) + 0.00000003f * get_color( src, green_t() ) + 1.32802486 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_C //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_C, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.05152202f * get_color( src, red_t() ) + 0.00000017f * get_color( src, green_t() ) + 0.00000018 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 0.97455227f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 1.10034835 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_C> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.95100260f * get_color( src, red_t() ) + 0.00000006f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 1.02611196f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000001f * get_color( src, green_t() ) + 0.90880305 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};



////////////////////////////////////////////////////////////////////////////////
// T_D50 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_D50, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.17609382f * get_color( src, red_t() ) + 0.00000007f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000006f * get_color( src, red_t() ) + 0.97570127f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 0.72197068 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_D50> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.85027254f * get_color( src, red_t() ) + 0.00000001f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000001f * get_color( src, red_t() ) + 1.02490389f * get_color( src, green_t() ) - 0.00000004 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000001f * get_color( src, green_t() ) + 1.38509774 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_D55 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_D55, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.10405362f * get_color( src, red_t() ) + 0.00000012f * get_color( src, green_t() ) + 0.00000006 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 0.98688960f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.82335168 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_D55> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.90575325f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 1.01328468f * get_color( src, green_t() ) - 0.00000004 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000001f * get_color( src, green_t() ) + 1.21454775 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_D58 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_D58, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.06815648f * get_color( src, red_t() ) + 0.00000016f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000004f * get_color( src, red_t() ) + 0.99201381f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 0.87833548 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_D58> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.93619251f * get_color( src, red_t() ) + 0.00000009f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000001f * get_color( src, red_t() ) + 1.00805032f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	= -0.00000001f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 1.13851726 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};




////////////////////////////////////////////////////////////////////////////////
// T_D65 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_D65, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  get_color( src, red_t  () );
		get_color( dst, green_t() )	=  get_color( src, green_t() );
		get_color( dst, blue_t()  )	=  get_color( src, blue_t () );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_D65> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  get_color( src, red_t  () );
		get_color( dst, green_t() )	=  get_color( src, green_t() );
		get_color( dst, blue_t()  )	=  get_color( src, blue_t () );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_D75 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_D75, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.92909920f * get_color( src, red_t() ) + 0.00000008f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 1.00641775f * get_color( src, green_t() ) + 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 1.14529049 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_D75> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.07631159f * get_color( src, red_t() ) + 0.00000002f * get_color( src, green_t() ) + 0.00000006 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000001f * get_color( src, red_t() ) + 0.99362320f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.87314081 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_9300 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_9300, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.89304549f * get_color( src, red_t() ) + 0.00000012f * get_color( src, green_t() ) + 0.00000012 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 0.99430132f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 1.37155378 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_9300> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.11976373f * get_color( src, red_t() ) - 0.00000004f * get_color( src, green_t() ) - 0.00000006 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000000f * get_color( src, red_t() ) + 1.00573123f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.72909999 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_E //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_E, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.20491767f * get_color( src, red_t() ) + 0.00000008f * get_color( src, green_t() ) + 0.00000012 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000003f * get_color( src, red_t() ) + 0.94827831f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.90876031 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_E> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.82993227f * get_color( src, red_t() ) - 0.00000005f * get_color( src, green_t() ) - 0.00000009 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000005f * get_color( src, red_t() ) + 1.05454266f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 1.10040021 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};


////////////////////////////////////////////////////////////////////////////////
// T_F2 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_F2, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.34113419f * get_color( src, red_t() ) + 0.00000007f * get_color( src, green_t() ) + 0.00000006 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000004f * get_color( src, red_t() ) + 0.94260973f * get_color( src, green_t() ) - 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 0.56362498 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_F2> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.74563771f * get_color( src, red_t() ) + 0.00000003f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000002f * get_color( src, red_t() ) + 1.06088448f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000003f * get_color( src, green_t() ) + 1.77422941 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};



////////////////////////////////////////////////////////////////////////////////
// T_F7 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_F7, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.00054073f * get_color( src, red_t() ) + 0.00000006f * get_color( src, green_t() ) + 0.00000012 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000003f * get_color( src, red_t() ) + 0.99999511f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.99845666 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_F7> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.99945968f * get_color( src, red_t() ) - 0.00000008f * get_color( src, green_t() ) - 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000001f * get_color( src, red_t() ) + 1.00000501f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 1.00154579 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};



////////////////////////////////////////////////////////////////////////////////
// T_F11 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_F11, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.20491767f * get_color( src, red_t() ) + 0.00000008f * get_color( src, green_t() ) + 0.00000012 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000003f * get_color( src, red_t() ) + 0.94827831f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.90876031 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_F11> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.70728123f * get_color( src, red_t() ) + 0.00000006f * get_color( src, green_t() ) + 0.00000006 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000002f * get_color( src, red_t() ) + 1.08209848f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	= -0.00000001f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 1.87809896 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}

};



////////////////////////////////////////////////////////////////////////////////
// T_DCPI3 //

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_DCIP3, temperature::T_INTER> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  0.88602084f * get_color( src, red_t() ) + 0.00000013f * get_color( src, green_t() ) + 0.00000003 * get_color( src, blue_t() );
		get_color( dst, green_t() )	= -0.00000003f * get_color( src, red_t() ) + 1.04855490f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) - 0.00000000f * get_color( src, green_t() ) + 0.85470724 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};

/**
 * @brief
 */
template< typename Channel >
struct channel_color_temperature_t<Channel, temperature::T_INTER, temperature::T_DCIP3> : public std::binary_function<Channel, Channel, Channel>
{
	typedef typename floating_channel_type_t<Channel>::type T;
	typedef typename channel_traits<Channel>::const_reference ChannelConstRef;
	typedef typename channel_traits<Channel>::reference ChannelRef;

	ChannelRef operator()( ChannelConstRef src, ChannelRef dst ) const
	{
		get_color( dst, red_t()   )	=  1.12864172f * get_color( src, red_t() ) - 0.00000001f * get_color( src, green_t() ) + 0.00000000 * get_color( src, blue_t() );
		get_color( dst, green_t() )	=  0.00000000f * get_color( src, red_t() ) + 0.95369333f * get_color( src, green_t() ) - 0.00000001 * get_color( src, blue_t() );
		get_color( dst, blue_t()  )	=  0.00000000f * get_color( src, red_t() ) + 0.00000000f * get_color( src, green_t() ) + 1.16999125 * get_color( src, blue_t() );
		copy_pixel_channel_if_exist<alpha_t>( src, dst );
		return dst;
	}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



template< typename Pixel,
          class IN,
          class OUT >
struct pixel_color_temperature_t
{
	typedef typename channel_type<Pixel>::type Channel;
	const IN&  _in;
	const OUT& _out;

	pixel_color_temperature_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	Pixel& operator()( const Pixel& p1,
	                   Pixel& p2 ) const
	{
		static_for_each(
				p1, p2,
				channel_color_temperature_t< Channel, IN, OUT >( _in, _out )
			);
		return p2;
	}
};

template< class IN,
          class OUT >
struct transform_pixel_color_temperature_t
{
	const IN&  _in;
	const OUT& _out;

	transform_pixel_color_temperature_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	template< typename Pixel>
	Pixel operator()( const Pixel& p1 ) const
	{
		Pixel p2;
		pixel_color_temperature_t<Pixel, IN, OUT>( _in, _out )( p1, p2 );
		return p2;
	}
};

/**
 * @example temperature_convert_view( srcView, dstView, temperature::sRGB(), temperature::Gamma(5.0) );
 */
template<class TemperatureIN, class TemperatureOUT, class View>
void temperature_convert_view( const View& src, View& dst, const TemperatureIN& temperatureIn = TemperatureIN(), const TemperatureOUT& temperatureOut = TemperatureOUT() )
{
	boost::gil::transform_pixels( src, dst, transform_pixel_color_temperature_t<TemperatureIN, TemperatureOUT>( temperatureIn, temperatureOut ), *this );
}

/**
 * @example temperature_convert_pixel( srcPix, dstPix, temperature::sRGB(), temperature::Gamma(5.0) );
 */
template<class TemperatureIN, class TemperatureOUT, class Pixel>
void temperature_convert_pixel( const Pixel& src, Pixel& dst, const TemperatureIN& temperatureIn = TemperatureIN(), const TemperatureOUT& temperatureOut = TemperatureOUT() )
{
	pixel_color_temperature_t<TemperatureIN, TemperatureOUT>( temperatureIn, temperatureOut )( src, dst );
}

}
}

#endif

