#ifndef _TERRY_ALGORITHM_STATIC_CHANNEL_RECURSION_HPP_
#define	_TERRY_ALGORITHM_STATIC_CHANNEL_RECURSION_HPP_

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/at.hpp>

#include <boost/gil/gil_config.hpp>
#include <boost/gil/gil_concept.hpp>
#include <boost/gil/utilities.hpp>

#include <algorithm>

namespace terry {
using namespace boost::gil;
namespace algorithm {

namespace ext_detail {

// compile-time recursion for per-element operations on color bases
template <int N>
struct element_recursion
{
    //static_for_each with four sources
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, P2& p2, P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, P2& p2, const P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, const P2& p2, P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, const P2& p2, const P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, P2& p2, P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, P2& p2, const P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, const P2& p2, P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, const P2& p2, const P3& p3, const P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }

    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, P2& p2, P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, P2& p2, const P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, const P2& p2, P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(P1& p1, const P2& p2, const P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, P2& p2, P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, P2& p2, const P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, const P2& p2, P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1& p1, const P2& p2, const P3& p3, P4& p4, Op op) {
        Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,p4,op));
        op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3), semantic_at_c<N-1>(p4));
        return op2;
    }

//    //static_transform with two sources
//    template <typename P1,typename P2,typename Dst,typename Op>
//    static Op static_transform(P1& src1, P2& src2, Dst& dst, Op op) {
//        Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
//        semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
//        return op2;
//    }
//    template <typename P1,typename P2,typename Dst,typename Op>
//    static Op static_transform(P1& src1, const P2& src2, Dst& dst, Op op) {
//        Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
//        semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
//        return op2;
//    }
//    template <typename P1,typename P2,typename Dst,typename Op>
//    static Op static_transform(const P1& src1, P2& src2, Dst& dst, Op op) {
//        Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
//        semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
//        return op2;
//    }
//    template <typename P1,typename P2,typename Dst,typename Op>
//    static Op static_transform(const P1& src1, const P2& src2, Dst& dst, Op op) {
//        Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
//        semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
//        return op2;
//    }
};

// Termination condition of the compile-time recursion for element operations on a color base
template<> struct element_recursion<0>
{
	//static_for_each with four sources
    template <typename P1,typename P2,typename P3,typename P4,typename Op>
    static Op static_for_each(const P1&,const P2&,const P3&,const P4&,Op op){return op;}

    //static_transform with three sources
    template <typename P1,typename P2,typename P3,typename Dst,typename Op>
    static Op static_transform(const P1&,const P2&,const P3&,const Dst&,Op op){return op;}
};

}


/**
\defgroup ColorBaseAlgorithmTransform static_transform
\ingroup ColorBaseAlgorithm
\brief Equivalent to std::transform. Pairs the elements semantically
*/

/// \{
////static_transform with three sources
//template <typename P2,typename P3,typename Dst,typename Op>
//GIL_FORCEINLINE
//Op static_transform(P2& p2,P3& p3,Dst& dst,Op op) { return ext_detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
//template <typename P2,typename P3,typename Dst,typename Op>
//GIL_FORCEINLINE
//Op static_transform(P2& p2,const P3& p3,Dst& dst,Op op) { return ext_detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
//template <typename P2,typename P3,typename Dst,typename Op>
//GIL_FORCEINLINE
//Op static_transform(const P2& p2,P3& p3,Dst& dst,Op op) { return ext_detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
//template <typename P2,typename P3,typename Dst,typename Op>
//GIL_FORCEINLINE
//Op static_transform(const P2& p2,const P3& p3,Dst& dst,Op op) { return ext_detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
/// \}

/**
\defgroup ColorBaseAlgorithmForEach static_for_each
\ingroup ColorBaseAlgorithm
\brief Equivalent to std::for_each. Pairs the elements semantically
*/

///\{
//static_for_each with four sources
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,P2& p2,P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,P2& p2,const P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2,P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2,const P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,P2& p2,P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,P2& p2,const P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2,P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2,const P3& p3,const P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }

template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,P2& p2,P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,P2& p2,const P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2,P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2,const P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,P2& p2,P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,P2& p2,const P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2,P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
template <typename P1,typename P2,typename P3,typename P4,typename Op>
GIL_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2,const P3& p3,P4& p4,Op op) { return ext_detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,p4,op); }
///\}

}
}


#endif
