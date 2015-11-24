#ifndef _CCTAG_SUBPIXELEDGEOPTIMIZER_HPP_
#define	_CCTAG_SUBPIXELEDGEOPTIMIZER_HPP_

#define SUBPIX_EDGE_OPTIM

#include <cctag/geometry/point.hpp>
#include <cctag/global.hpp>

#ifdef WITH_OPTPP
#include <OPT++_config.h>
#include <newmat.h>
#include <OptLBFGS.h>
#include <NLF.h>
#include <OptQNewton.h>
#include <newmat.h>
#endif

namespace cctag {

#if defined(WITH_OPTPP) && defined(SUBPIX_EDGE_OPTIM) // undefined. Depreciated

struct ImageCut;

/**
 * @brief Optimizer to find the subpixel position of an edge point from a 1D signal.
 */
class SubPixEdgeOptimizer : public OPTPP::FDNLF1
{
public:
	typedef SubPixEdgeOptimizer This;
	typedef OPTPP::FDNLF1 Parent;

public:
	SubPixEdgeOptimizer( const cctag::ImageCut & line );

	/**
	 * @brief Do optimization.
	 *
	 * All pararameters are used to initialize the optimization.
	 * @param[in] widthContour initial slope width
	 * @param[in] xp initial position
	 * @param[in] imin initial minimum signal
	 * @param[in] imax initial maximum signal
	 */
	Point2dN<double> operator()(const double widthContour, const double xp, const double imin, const double imax);

	inline void initFcn()
	{
		if ( init_flag == false )
		{
			initSubPix( dim, mem_xc );
			init_flag = true;
		}
		else
		{
			CCTAG_COUT_ERROR( "FDNLF1:initFcn: Warning - initialization called twice\n" );
			initSubPix( dim, mem_xc );
		}
	}

private:
	/// @brief Optimization initialization function.
	void initSubPix( int ndim, NEWMAT::ColumnVector& x );
	/// @brief Optimization cost function.
	static void subPix( int n, const NEWMAT::ColumnVector& x, double& fx, int& result, void* );

private:
	const cctag::ImageCut & _line;
	double _a, _b;
	double _widthContour;
	double _xp;
	double _imin;
	double _imax;
};

#endif // SUBPIX_EDGE_OPTIM // undefined. Depreciated

} // namespace cctag

#endif

