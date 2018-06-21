/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_SUBPIXELEDGEOPTIMIZER_HPP_
#define	_CCTAG_SUBPIXELEDGEOPTIMIZER_HPP_

#define SUBPIX_EDGE_OPTIM

#include <cctag/geometry/Point.hpp>
#include <cctag/utils/Defines.hpp>

namespace cctag {

#if defined(WITH_OPTPP) && defined(SUBPIX_EDGE_OPTIM) // undefined. Deprecated

struct ImageCut;

/**
 * @brief Optimizer to find the subpixel position of an edge point from a 1D signal.
 */
class SubPixEdgeOptimizer : public OPTPP::FDNLF1
{
public:
	using This = SubPixEdgeOptimizer;
	using Parent =OPTPP::FDNLF1;

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
	Point2d<Eigen::Vector3f> operator()(const float widthContour, const float xp, const float imin, const float imax);

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
	float _a, _b;
	float _widthContour;
	float _xp;
	float _imin;
	float _imax;
};

#endif // SUBPIX_EDGE_OPTIM // undefined. Depreciated

} // namespace cctag

#endif

