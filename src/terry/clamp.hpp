#ifndef _TERRY_CLAMP_HPP_
#define _TERRY_CLAMP_HPP_

namespace terry {

template <typename SChannel, typename DChannel>
struct channel_clamp
{
	const SChannel _min;
	const SChannel _max;
	
	channel_clamp( const SChannel min, const SChannel max )
	: _min(min)
	, _max(max)
	{}

	GIL_FORCEINLINE
	void operator()(const SChannel& src, DChannel& dst) const
	{
		if( src > _max )
			dst = _max;
		else if( src < _min )
			dst = _min;
		else
			dst = src;
	}
};

template<typename SPixel, typename DPixel = SPixel>
struct pixel_clamp
{
	typedef typename boost::gil::channel_type<SPixel>::type SChannel;
	typedef typename boost::gil::channel_type<DPixel>::type DChannel;
	
	const SChannel _min;
	const SChannel _max;
	
	pixel_clamp()
	: _min( boost::gil::channel_traits<SChannel>::min_value() )
	, _max( boost::gil::channel_traits<SChannel>::max_value() )
	{}

	pixel_clamp( const SChannel min, const SChannel max )
	: _min(min)
	, _max(max)
	{}

	GIL_FORCEINLINE
	void operator()( const SPixel& src, DPixel& dst ) const
	{
		using namespace boost::gil;
		static_for_each(src,dst,channel_clamp<SChannel, DChannel>(_min,_max));
	}
};

template <typename View>
GIL_FORCEINLINE
typename boost::gil::color_converted_view_type<View, typename View::value_type, pixel_clamp<typename View::value_type> >::type clamp_view( const View& sView )
{
	using namespace boost::gil;
	typedef typename View::value_type Pixel;
	return color_converted_view<Pixel>( sView, pixel_clamp<Pixel>() );
}

}

#endif

