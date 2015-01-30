#ifndef _TERRY_COLOR_COLORSPACE_BASE_HPP_
#define	_TERRY_COLOR_COLORSPACE_BASE_HPP_

#include <boost/mpl/vector.hpp>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/pixel.hpp>
#include <boost/gil/typedefs.hpp>

#define TERRY_DEFINE_GIL_INTERNALS_4(CS) \
	\
	template <typename IC> \
	inline typename type_from_x_iterator<planar_pixel_iterator<IC,color::CS##_colorspace_t> >::view_t \
	planar_##CS##_view(std::size_t width, std::size_t height, IC a, IC b, IC c, IC d, std::ptrdiff_t rowsize_in_bytes) { \
		typedef typename type_from_x_iterator<planar_pixel_iterator<IC,color::CS##_colorspace_t> >::view_t RView; \
		return RView(width, height, typename RView::locator(planar_pixel_iterator<IC,color::CS##_colorspace_t>(a,b,c,d), rowsize_in_bytes)); \
	}


#define TERRY_DEFINE_GIL_INTERNALS_3(CS) \
	\
	template <typename IC> \
	inline typename type_from_x_iterator<planar_pixel_iterator<IC,color::CS##_colorspace_t> >::view_t \
	planar_##CS##_view(std::size_t width, std::size_t height, \
					IC a, IC b, IC c, \
					std::ptrdiff_t rowsize_in_bytes) { \
		typedef typename type_from_x_iterator<planar_pixel_iterator<IC,color::CS##_colorspace_t> >::view_t RView; \
		return RView(width, height, \
					 typename RView::locator(planar_pixel_iterator<IC,color::CS##_colorspace_t>(a,b,c), \
											 rowsize_in_bytes)); \
	}


#define TERRY_DEFINE_ALL_TYPEDEFS(T,CS)         \
    GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(T,CS,color::CS##_colorspace_t,color::CS##_layout_t)

#define TERRY_DEFINE_COLORSPACE_STANDARD_TYPEDEFS(CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(8,  CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(8s, CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(16, CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(16s,CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(32 ,CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(32s,CS) \
	TERRY_DEFINE_ALL_TYPEDEFS(32f,CS)

namespace terry {
using namespace ::boost::gil;

using ::boost::gil::pixel;
using ::boost::gil::planar_pixel_reference;
using ::boost::gil::planar_pixel_iterator;
using ::boost::gil::memory_based_step_iterator;
using ::boost::gil::point2;
using ::boost::gil::memory_based_2d_locator;
using ::boost::gil::image_view;
using ::boost::gil::image;

namespace color {

/**
 * @brief Base class of all color parameters class.
 */
struct IColorParams
{
	virtual bool operator==( const IColorParams& other ) const = 0;
	bool operator!=( const IColorParams& other ) const { return ! this->operator==( other ); };
};


/**
 * @brief Fake class to finish hierachy.
 */
struct None {};
/**
 * @brief Fake class to finish hierachy.
 */
struct IsRootReference
{
	typedef None reference;
	typedef None params;
};


}
}

#endif

