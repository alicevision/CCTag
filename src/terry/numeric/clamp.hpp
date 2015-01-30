#ifndef _TERRY_NUMERIC_CLAMP_HPP_
#define _TERRY_NUMERIC_CLAMP_HPP_


namespace terry {
namespace numeric {

template <typename Channel>
struct channel_clamp_lower_than_t : public std::unary_function<Channel, Channel>
{
	Channel _threshold;
	Channel _replace;

	channel_clamp_lower_than_t( const Channel threshold, const Channel replace )
	: _threshold( threshold )
	, _replace( replace )
	{ }

	GIL_FORCEINLINE
	Channel operator( )( typename channel_traits<Channel>::const_reference ch ) const
	{
		if( ch < _threshold )
			return _replace;
		return Channel( ch );
	}
};

template <typename Pixel> // models pixel concept
struct pixel_clamp_lower_than_t
{
	typedef typename channel_type<Pixel>::type Channel;
	Channel _threshold;
	Channel _replace;

	pixel_clamp_lower_than_t( const Channel threshold, const Channel replace )
	: _threshold( threshold )
	, _replace( replace )
	{}

	GIL_FORCEINLINE
	Pixel operator()( const Pixel & p ) const
	{
		Pixel result;
		static_transform( p, result, channel_clamp_lower_than_t<typename channel_type<Pixel>::type>( _threshold, _replace ) );
		return result;
	}
};


}
}

#endif
