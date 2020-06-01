/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "Operation.hpp"

#include <Eigen/LU>

#include <cmath>

namespace cctag {
namespace numerical {

Eigen::Matrix3f& normalizeDet1( Eigen::Matrix3f& m )
{

	const float det = m.determinant();
	if( det == 0 )
		return m;

	const float s = ( ( det >= 0 ) ? 1 : -1 ) / std::pow( std::abs( det ), 1.f / 3.f );
	m = s * m;
	return m;
}

} // numerical
} // cctag