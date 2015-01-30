#ifndef _GIL_EXTENSION_TYPEDEFS_HPP_
#define _GIL_EXTENSION_TYPEDEFS_HPP_

#include <boost/integer.hpp>  // for boost::uint_t
#include <boost/gil/gil_all.hpp>
#include <boost/type_traits.hpp>

#define I10_MIN 0               // 0
#define I10_MAX 1023            // 2^10 - 1

namespace boost {
namespace gil  {

///////////////////////////////////////////
////
////  Add built-in channel models for 64bits
////
///////////////////////////////////////////

/// \defgroup bits64 bits64
/// \ingroup ChannelModel
/// \brief 64-bit unsigned integral channel type  (typedef from uint64_t). Models ChannelValueConcept

/// \ingroup bits64
typedef uint64_t bits64;

/// \defgroup bits64s bits64s
/// \ingroup ChannelModel
/// \brief 64-bit signed integral channel type (typedef from int64_t). Models ChannelValueConcept

/// \ingroup bits64s
typedef int64_t bits64s;

struct double_zero
{
	static double apply() { return 0.0; }
};
struct double_one
{
	static double apply() { return 1.0; }
};

/// \defgroup bits64f bits64f
/// \ingroup ChannelModel
/// \brief 64-bit floating point channel type with range [0.0f ... 1.0f]. Models ChannelValueConcept

/// \ingroup bits64f
typedef scoped_channel_value<double, double_zero, double_one> bits64f;

GIL_DEFINE_BASE_TYPEDEFS( 64, gray )
GIL_DEFINE_BASE_TYPEDEFS( 64s, gray )
GIL_DEFINE_BASE_TYPEDEFS( 64f, gray )
GIL_DEFINE_BASE_TYPEDEFS( 64, bgr )
GIL_DEFINE_BASE_TYPEDEFS( 64s, bgr )
GIL_DEFINE_BASE_TYPEDEFS( 64f, bgr )
GIL_DEFINE_BASE_TYPEDEFS( 64, argb )
GIL_DEFINE_BASE_TYPEDEFS( 64s, argb )
GIL_DEFINE_BASE_TYPEDEFS( 64f, argb )
GIL_DEFINE_BASE_TYPEDEFS( 64, abgr )
GIL_DEFINE_BASE_TYPEDEFS( 64s, abgr )
GIL_DEFINE_BASE_TYPEDEFS( 64f, abgr )
GIL_DEFINE_BASE_TYPEDEFS( 64, bgra )
GIL_DEFINE_BASE_TYPEDEFS( 64s, bgra )
GIL_DEFINE_BASE_TYPEDEFS( 64f, bgra )

GIL_DEFINE_ALL_TYPEDEFS( 64, rgb )
GIL_DEFINE_ALL_TYPEDEFS( 64s, rgb )
GIL_DEFINE_ALL_TYPEDEFS( 64f, rgb )
GIL_DEFINE_ALL_TYPEDEFS( 64, rgba )
GIL_DEFINE_ALL_TYPEDEFS( 64s, rgba )
GIL_DEFINE_ALL_TYPEDEFS( 64f, rgba )
GIL_DEFINE_ALL_TYPEDEFS( 64, cmyk )
GIL_DEFINE_ALL_TYPEDEFS( 64s, cmyk )
GIL_DEFINE_ALL_TYPEDEFS( 64f, cmyk )

template <int N>
struct devicen_t;
template <int N>
struct devicen_layout_t;
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64, dev2n, devicen_t<2>, devicen_layout_t<2>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64s, dev2n, devicen_t<2>, devicen_layout_t<2>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64f, dev2n, devicen_t<2>, devicen_layout_t<2>)

GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64, dev3n, devicen_t<3>, devicen_layout_t<3>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64s, dev3n, devicen_t<3>, devicen_layout_t<3>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64f, dev3n, devicen_t<3>, devicen_layout_t<3>)

GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64, dev4n, devicen_t<4>, devicen_layout_t<4>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64s, dev4n, devicen_t<4>, devicen_layout_t<4>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64f, dev4n, devicen_t<4>, devicen_layout_t<4>)

GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64, dev5n, devicen_t<5>, devicen_layout_t<5>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64s, dev5n, devicen_t<5>, devicen_layout_t<5>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 64f, dev5n, devicen_t<5>, devicen_layout_t<5>)


/******************************************************************************
 * Bit stream
 *****************************************************************************/
typedef uint16_t uint10_t;
typedef uint16_t uint12_t;

/// 10 bits rgb bit stream
typedef bit_aligned_pixel_reference< uint32_t,
                                     mpl::vector3_c<uint10_t, 10, 10, 10>,
                                     rgb_layout_t,
                                     true
                                     >  rgb10_stream_ref_t;
typedef bit_aligned_pixel_iterator<rgb10_stream_ref_t> rgb10_stream_ptr_t;
typedef std::iterator_traits<rgb10_stream_ptr_t>::value_type rgb10_stream_pixel_t;
typedef view_type_from_pixel<rgb10_stream_pixel_t>::type rgb10_stream_view_t;

/// 10 bits rgba bit stream
typedef bit_aligned_pixel_reference< uint64_t,
                                     mpl::vector4_c<uint10_t, 10, 10, 10, 10>,
                                     rgba_layout_t,
                                     true
                                     >  rgba10_stream_ref_t;
typedef bit_aligned_pixel_iterator<rgba10_stream_ref_t> rgba10_stream_ptr_t;
typedef std::iterator_traits<rgba10_stream_ptr_t>::value_type rgba10_stream_pixel_t;
typedef view_type_from_pixel<rgba10_stream_pixel_t>::type rgba10_stream_view_t;

/// 10 bits abgr bit stream
typedef bit_aligned_pixel_reference< uint64_t,
                                     mpl::vector4_c<uint10_t, 10, 10, 10, 10>,
                                     abgr_layout_t,
                                     true
                                     >  abgr10_stream_ref_t;
typedef bit_aligned_pixel_iterator<abgr10_stream_ref_t> abgr10_stream_ptr_t;
typedef std::iterator_traits<abgr10_stream_ptr_t>::value_type abgr10_stream_pixel_t;
typedef view_type_from_pixel<abgr10_stream_pixel_t>::type abgr10_stream_view_t;

/// 12 bits rgb bit stream
typedef bit_aligned_pixel_reference< uint64_t,
                                     mpl::vector3_c<uint12_t, 12, 12, 12>,
                                     rgb_layout_t,
                                     true
                                     >  rgb12_stream_ref_t;
typedef bit_aligned_pixel_iterator<rgb12_stream_ref_t> rgb12_stream_ptr_t;
typedef std::iterator_traits<rgb12_stream_ptr_t>::value_type rgb12_stream_pixel_t;
typedef view_type_from_pixel<rgb12_stream_pixel_t>::type rgb12_stream_view_t;

/// 12 bits rgba bit stream
typedef bit_aligned_pixel_reference< uint64_t,
                                     mpl::vector4_c<uint12_t, 12, 12, 12, 12>,
                                     rgba_layout_t,
                                     true
                                     >  rgba12_stream_ref_t;
typedef bit_aligned_pixel_iterator<rgba12_stream_ref_t> rgba12_stream_ptr_t;
typedef std::iterator_traits<rgba12_stream_ptr_t>::value_type rgba12_stream_pixel_t;
typedef view_type_from_pixel<rgba12_stream_pixel_t>::type rgba12_stream_view_t;

/// 12 bits abgr bit stream
typedef bit_aligned_pixel_reference< uint64_t,
                                     mpl::vector4_c<uint12_t, 12, 12, 12, 12>,
                                     abgr_layout_t,
                                     true
                                     >  abgr12_stream_ref_t;
typedef bit_aligned_pixel_iterator<abgr12_stream_ref_t> abgr12_stream_ptr_t;
typedef std::iterator_traits<abgr12_stream_ptr_t>::value_type abgr12_stream_pixel_t;
typedef view_type_from_pixel<abgr12_stream_pixel_t>::type abgr12_stream_view_t;

/// 12 bits rgba packed to short
typedef const packed_channel_reference<uint64_t, 04, 12, true> rgba12_packed_channel0_t;
typedef const packed_channel_reference<uint64_t, 20, 12, true> rgba12_packed_channel1_t;
typedef const packed_channel_reference<uint64_t, 36, 12, true> rgba12_packed_channel2_t;
typedef const packed_channel_reference<uint64_t, 52, 12, true> rgba12_packed_channel3_t;
typedef mpl::vector4<rgba12_packed_channel0_t, rgba12_packed_channel1_t,
                     rgba12_packed_channel2_t, rgba12_packed_channel3_t> rgba12_packed_channels_t;
typedef packed_pixel<uint64_t, rgba12_packed_channels_t, rgba_layout_t> rgba12_packed_pixel_t;
typedef view_type_from_pixel<rgba12_packed_pixel_t>::type rgba12_packed_view_t;
typedef image<rgba12_packed_pixel_t, false> rgba12_packed_image_t;

/// 12 bits abgr packed to short
typedef const packed_channel_reference<uint64_t, 04, 12, true> abgr12_packed_channel0_t;
typedef const packed_channel_reference<uint64_t, 20, 12, true> abgr12_packed_channel1_t;
typedef const packed_channel_reference<uint64_t, 36, 12, true> abgr12_packed_channel2_t;
typedef const packed_channel_reference<uint64_t, 52, 12, true> abgr12_packed_channel3_t;
typedef mpl::vector4<abgr12_packed_channel0_t, abgr12_packed_channel1_t,
                     abgr12_packed_channel2_t, abgr12_packed_channel3_t> abgr12_packed_channels_t;
typedef packed_pixel<uint64_t, abgr12_packed_channels_t, abgr_layout_t> abgr12_packed_pixel_t;
typedef view_type_from_pixel<abgr12_packed_pixel_t>::type abgr12_packed_view_t;
typedef image<abgr12_packed_pixel_t, false> abgr12_packed_image_t;

/// 12 bits rgb packed to 6 bytes
typedef const packed_channel_reference<uint64_t, 04, 12, true> rgb12_packed_channel0_t;
typedef const packed_channel_reference<uint64_t, 20, 12, true> rgb12_packed_channel1_t;
typedef const packed_channel_reference<uint64_t, 36, 12, true> rgb12_packed_channel2_t;
typedef mpl::vector3<rgb12_packed_channel0_t, rgb12_packed_channel1_t, rgb12_packed_channel2_t> rgb12_packed_channels_t;
typedef packed_pixel<packed_channel_value<48>, rgb12_packed_channels_t, rgb_layout_t> rgb12_packed_pixel_t;
typedef view_type_from_pixel<rgb12_packed_pixel_t>::type rgb12_packed_view_t;
typedef image<rgb12_packed_pixel_t, false> rgb12_packed_image_t;

/// 10 bits rgb packed to uint32_t
typedef const packed_channel_reference<uint32_t, 22, 10, true> rgb10_packed_channel0_t;
typedef const packed_channel_reference<uint32_t, 12, 10, true> rgb10_packed_channel1_t;
typedef const packed_channel_reference<uint32_t, 02, 10, true> rgb10_packed_channel2_t;
typedef mpl::vector3<rgb10_packed_channel0_t, rgb10_packed_channel1_t, rgb10_packed_channel2_t> rgb10_packed_channels_t;
typedef packed_pixel<uint32_t, rgb10_packed_channels_t, rgb_layout_t> rgb10_packed_pixel_t;
typedef view_type_from_pixel<rgb10_packed_pixel_t>::type rgb10_packed_view_t;
typedef image<rgb10_packed_pixel_t, false> rgb10_packed_image_t;

/// 10 bits rgba packed to short
typedef const packed_channel_reference<uint64_t, 22, 10, true> rgba10_packed_channel0_t;
typedef const packed_channel_reference<uint64_t, 12, 10, true> rgba10_packed_channel1_t;
typedef const packed_channel_reference<uint64_t, 02, 10, true> rgba10_packed_channel2_t;
typedef const packed_channel_reference<uint64_t, 54, 10, true> rgba10_packed_channel3_t;
typedef mpl::vector4<rgba10_packed_channel0_t, rgba10_packed_channel1_t,
                     rgba10_packed_channel2_t, rgba10_packed_channel3_t> rgba10_packed_channels_t;
typedef packed_pixel<uint64_t, rgba10_packed_channels_t, rgba_layout_t> rgba10_packed_pixel_t;
typedef view_type_from_pixel<rgba10_packed_pixel_t>::type rgba10_packed_view_t;
typedef image<rgba10_packed_pixel_t, false> rgba10_packed_image_t;

/// 10 bits abgr packed to short
typedef const packed_channel_reference<uint64_t, 4, 10, true> abgr10_packed_channel0_t;
typedef const packed_channel_reference<uint64_t, 20, 10, true> abgr10_packed_channel1_t;
typedef const packed_channel_reference<uint64_t, 36, 10, true> abgr10_packed_channel2_t;
typedef const packed_channel_reference<uint64_t, 52, 10, true> abgr10_packed_channel3_t;
typedef mpl::vector4<abgr10_packed_channel0_t, abgr10_packed_channel1_t,
                     abgr10_packed_channel2_t, abgr10_packed_channel3_t> abgr10_packed_channels_t;
typedef packed_pixel<uint64_t, abgr10_packed_channels_t, abgr_layout_t> abgr10_packed_pixel_t;
typedef view_type_from_pixel<abgr10_packed_pixel_t>::type abgr10_packed_view_t;
typedef image<abgr10_packed_pixel_t, false> abgr10_packed_image_t;

/******************************************************************************
 * Packed on bytes view types
 *****************************************************************************/
typedef packed_channel_value<10> bits10;
typedef packed_channel_value<12> bits12;

GIL_DEFINE_BASE_TYPEDEFS( 10, gray )
GIL_DEFINE_BASE_TYPEDEFS( 10, bgr )
GIL_DEFINE_BASE_TYPEDEFS( 10, argb )
GIL_DEFINE_BASE_TYPEDEFS( 10, bgra )
GIL_DEFINE_BASE_TYPEDEFS( 10, abgr )
GIL_DEFINE_ALL_TYPEDEFS( 10, rgb )
GIL_DEFINE_ALL_TYPEDEFS( 10, rgba )
GIL_DEFINE_ALL_TYPEDEFS( 10, cmyk )

GIL_DEFINE_BASE_TYPEDEFS( 12, gray )
GIL_DEFINE_BASE_TYPEDEFS( 12, bgr )
GIL_DEFINE_BASE_TYPEDEFS( 12, argb )
GIL_DEFINE_BASE_TYPEDEFS( 12, bgra )
GIL_DEFINE_BASE_TYPEDEFS( 12, abgr )
GIL_DEFINE_ALL_TYPEDEFS( 12, rgb )
GIL_DEFINE_ALL_TYPEDEFS( 12, rgba )
GIL_DEFINE_ALL_TYPEDEFS( 12, cmyk )

GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 10, dev2n, devicen_t<2>, devicen_layout_t<2>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 10, dev3n, devicen_t<3>, devicen_layout_t<3>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 10, dev4n, devicen_t<4>, devicen_layout_t<4>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 10, dev5n, devicen_t<5>, devicen_layout_t<5>)

GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 12, dev2n, devicen_t<2>, devicen_layout_t<2>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 12, dev3n, devicen_t<3>, devicen_layout_t<3>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 12, dev4n, devicen_t<4>, devicen_layout_t<4>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL( 12, dev5n, devicen_t<5>, devicen_layout_t<5>)

}
}

#endif

