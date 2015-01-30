#ifndef _TERRY_PIXELPROXY_HPP_
#define _TERRY_PIXELPROXY_HPP_

#include <boost/type_traits/remove_reference.hpp>

namespace terry {

/// \brief Returns the reference proxy associated with a type that has a \p "reference" member typedef.
///
/// The reference proxy is the reference type, but with stripped-out C++ reference. It models PixelConcept
template <typename T>
struct pixel_proxy : public ::boost::remove_reference<typename T::reference> {};

}

#endif
