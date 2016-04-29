#ifndef _CCTAG_2DTRANSFORM_HPP_
#define _CCTAG_2DTRANSFORM_HPP_

#include <vector>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/algebra/matrix/Operation.hpp>
#include <boost/foreach.hpp>
#include <Eigen/Core>

namespace cctag {
namespace viewGeometry {

template<class C>
inline void projectiveTransform( const Eigen::Matrix3f& tr, std::vector<C>& v )
{
  for (const auto& p : v) {
    auto pf = p.template cast<float>();
    C ptAux = tr * pf;
    p = cctag::numerical::normalize( ptAux );
  }
}

inline void projectiveTransform( const Eigen::Matrix3f& tr, cctag::numerical::geometry::Ellipse& ellipse )
{
  auto m = tr.transpose() * ellipse.matrix() * tr;
  ellipse.setMatrix(m);
	//ellipse.setMatrix(
        //ublas::prec_prod(
        //    ublas::trans( tr ),
        //    (ublas::bounded_matrix<double, 3, 3>)ublas::prec_prod(
        //        ellipse.matrix(),
        //        tr ) ) );
}

inline void projectiveTransform( const Eigen::Matrix3f& tr, const Eigen::Matrix3f& ttr, cctag::numerical::geometry::Ellipse& ellipse )
{
  auto m = ttr * ellipse.matrix() * tr;
  ellipse.setMatrix(m);
	//ellipse.setMatrix( ublas::prec_prod( ttr, ( ublas::bounded_matrix<double, 3, 3>) ublas::prec_prod( ellipse.matrix(), tr ) ) );
}

} // namespace viewGeometry
} // namespace cctag

#endif

