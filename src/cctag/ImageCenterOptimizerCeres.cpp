#ifdef USE_IMAGE_CENTER_OPT_CERES // undefined. Depreciated

#include <cctag/ImageCenterOptimizerCeres.hpp>
#include <cctag/visualDebug.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/progBase/exceptions.hpp>
#include <cctag/global.hpp>

#include <terry/sampler/all.hpp>

#include <boost/bind.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <ostream>

namespace cctag {
namespace identification {


void optimizeCenterCeres(cctag::Point2dN<double> initCenter, const TotoFunctor::VecExtPoints & vecExtPoints, const std::size_t lengthSig, const boost::gil::gray8_view_t & sView,
  const cctag::numerical::geometry::Ellipse & outerEllipse){
// The variable to solve for with its initial value. It will be
  // mutated in place by the solver.

//cctag::Point2dN<double> initCenter(400,400);

	std::vector<double> x;
	//x.reserve(2);

	x[0] = initCenter.x();
	x[1] = initCenter.y();

  //double x = 0.5;
  //const double initial_x = x;

  // Build the problem.
  ceres::Problem problem;

  /*

  // Set up the only cost function (also known as residual). This uses
  // numeric differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function =
      new ceres::NumericDiffCostFunction<TotoFunctor, ceres::CENTRAL, 1, 2> (new TotoFunctor(vecExtPoints, lengthSig, sView, outerEllipse )   );
  problem.AddResidualBlock(cost_function, NULL, &x[0]);

  // Run the solver!
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initCenter
            << " -> " << (new Point2dN<double>(x[0],x[1])) << "\n";
*/
  return;
}

} // namespace identification
} // namespace cctag

#endif // USE_IMAGE_CENTER_OPT_CERES // undefined. Depreciated