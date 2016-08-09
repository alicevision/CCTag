#ifdef USE_IMAGE_CENTER_OPT_MINPACK // undefined. Depreciated

#ifndef _CCTAG_LMIMAGECENTEROPTIMIZER_HPP_
#define	_CCTAG_LMIMAGECENTEROPTIMIZER_HPP_

#ifdef WITH_CMINPACK

#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>

#include <cctag/CCTag.hpp>

namespace cctag {

class LMImageCenterOptimizer
{
public:
	typedef std::vector< cctag::Point2d<Eigen::Vector3f> > VecExtPoints;
public:
	LMImageCenterOptimizer();
	virtual ~LMImageCenterOptimizer();
	/**
	 * @brief Do optimization.
	 *
	 * @param[in] cctagToRefine initial cctag to refine
	 * @return residual error
	 */
	float operator()( CCTag & cctagToRefine );
	static int homology( void* p, int m, int n, const float* x, float* fvec, int iflag );

private:
	cctag::Point2d<Eigen::Vector3f> _pToRefine;
};

} // namespace cctag

#endif

#endif

#endif //USE_IMAGE_CENTER_OPT_MINPACK // undefined. Depreciated
