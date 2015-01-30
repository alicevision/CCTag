/*
    Copyright 2005-2007 Adobe Systems Incorporated
    Distributed under the MIT License (see accompanying file LICENSE_1_0_0.txt
    or a copy at http://opensource.adobe.com/licenses.html)
*/

/*************************************************************************************************/

#ifndef _TERRY_FILTER_INNERPRODUCT_HPP_
#define _TERRY_FILTER_INNERPRODUCT_HPP_

/*!
/// \file               
/// \brief Numeric algorithms
/// \author Hailin Jin and Lubomir Bourdev \n
///         Adobe Systems Incorporated
/// \date   2005-2007 \n Last updated on February 6, 2007
*/

#include <terry/pixel_proxy.hpp>

#include <boost/gil/gil_config.hpp>
#include <boost/gil/pixel_iterator.hpp>
#include <boost/gil/metafunctions.hpp>

#include <cassert>
#include <iterator>
#include <algorithm>
#include <numeric>

namespace terry {

using namespace boost::gil;
namespace filter {

namespace detail {
template <std::size_t Size>
struct inner_product_k_t
{
    template <class _InputIterator1, class _InputIterator2, class _Tp,
              class _BinaryOperation1, class _BinaryOperation2>
	GIL_FORCEINLINE
    static _Tp apply( _InputIterator1 __first1, 
                      _InputIterator2 __first2, _Tp __init, 
                      _BinaryOperation1 __binary_op1,
                      _BinaryOperation2 __binary_op2)
	{
        __init = __binary_op1(__init, __binary_op2(*__first1, *__first2));
        return inner_product_k_t<Size-1>::template apply(__first1+1,__first2+1,__init,
                                                         __binary_op1, __binary_op2);
    }
};

template <>
struct inner_product_k_t<0>
{
    template <class _InputIterator1, class _InputIterator2, class _Tp,
              class _BinaryOperation1, class _BinaryOperation2>
	GIL_FORCEINLINE
    static _Tp apply( _InputIterator1 __first1, 
                      _InputIterator2 __first2, _Tp __init, 
                      _BinaryOperation1 __binary_op1,
                      _BinaryOperation2 __binary_op2 )
	{
        return __init;
    }
};
} // namespace detail

/// static version of std::inner_product
template <std::size_t Size,
          class _InputIterator1, class _InputIterator2, class _Tp,
          class _BinaryOperation1, class _BinaryOperation2>
GIL_FORCEINLINE
_Tp inner_product_k( _InputIterator1 __first1, 
                     _InputIterator2 __first2,
                     _Tp __init, 
                     _BinaryOperation1 __binary_op1,
                     _BinaryOperation2 __binary_op2 )
{
    return detail::inner_product_k_t<Size>::template apply(__first1,__first2,__init,
                                                           __binary_op1, __binary_op2);
}


}
}


#endif

