/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Point.hpp>
// #include <cctag/algebra/Invert.hpp>
#include <cctag/utils/Exceptions.hpp>
#include <cctag/utils/Defines.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>

namespace cctag {
namespace numerical {
namespace geometry {

Ellipse::Ellipse( const Eigen::Matrix3f& matrix )
{
	_matrix = matrix;
	computeParameters();
}

Ellipse::Ellipse( const Point2d<Eigen::Vector3f>& center, float a, float b, float angle )
{
	init( center, a, b, angle );
}

void Ellipse::init( const Point2d<Eigen::Vector3f>& center, float a, float b, float angle )
{
	if( a < 0.f || b < 0.f )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}

	_center = center;
	_a      = a;
	_b      = b;
	_angle  = angle;

	computeMatrix();
}

void Ellipse::setMatrix( const Eigen::Matrix3f& matrix )
{
	_matrix = matrix;
	computeParameters();
}

void Ellipse::setParameters( const Point2d<Eigen::Vector3f>& center, float a, float b, float angle )
{
	if( a < 0.f || b < 0.f )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}
	_center = center;
	_a      = a;
	_b      = b;
	_angle  = angle;
	computeMatrix();
}

void Ellipse::setCenter( const Point2d<Eigen::Vector3f>& center )
{
	_center = center;
	computeMatrix();
}

void Ellipse::setA( float a )
{
	if( a < 0.f )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}
	_a = a;
	computeMatrix();
}

void Ellipse::setB( float b )
{
	if( b < 0.f )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}
	_b = b;
	computeMatrix();
}

void Ellipse::setAngle( float angle )
{
	_angle = angle;
	computeMatrix();
}

Ellipse Ellipse::transform(const Matrix& mT) const
{
  auto a = mT.transpose() * _matrix;
  auto mET = a * mT;
  //const Matrix a = prec_prod( boost::numeric::ublas::trans(mT), _matrix );
  //const Matrix mET = prec_prod( a, mT );
  return Ellipse( mET );
}

void Ellipse::computeParameters()
{
        Eigen::VectorXf par(6);
	par( 0 ) = _matrix( 0, 0 );
	par( 1 ) = 2.f * _matrix( 0, 1 );
	par( 2 ) = _matrix( 1, 1 );
	par( 3 ) = 2.f * _matrix( 0, 2 );
	par( 4 ) = 2.f * _matrix( 1, 2 );
	par( 5 ) = _matrix( 2, 2 );

	const float thetarad    = 0.5f * std::atan2( par( 1 ), par( 0 ) - par( 2 ) );
	const float cost        = std::cos( thetarad );
	const float sint        = std::sin( thetarad );
	const float sin_squared = sint * sint;
	const float cos_squared = cost * cost;
	const float cos_sin     = sint * cost;

	const float Ao  = par( 5 );
	const float Au  = par( 3 ) * cost + par( 4 ) * sint;
	const float Av  = -par( 3 ) * sint + par( 4 ) * cost;
	const float Auu = par( 0 ) * cos_squared + par( 2 ) * sin_squared + par( 1 ) * cos_sin;
	const float Avv = par( 0 ) * sin_squared + par( 2 ) * cos_squared - par( 1 ) * cos_sin;

	if( Auu == 0 || Avv == 0 )
	{
		_center = Point2d<Eigen::Vector3f>( 0.f, 0.f );
		_a      = 0.f;
		_b      = 0.f;
		_angle  = 0.f;
	}
	else
	{
		const float tuCentre = -Au / ( 2.0f * Auu );
		const float tvCentre = -Av / ( 2.0f * Avv );
		const float wCentre  = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;

		_center = Point2d<Eigen::Vector3f>( tuCentre * cost - tvCentre * sint, tuCentre * sint + tvCentre * cost );

		const float Ru = -wCentre / Auu;
		const float Rv = -wCentre / Avv;

		const float aAux = std::sqrt( std::abs( Ru ) ) * boost::math::sign( Ru );
		const float bAux = std::sqrt( std::abs( Rv ) ) * boost::math::sign( Rv );

		if( aAux < 0.f || bAux < 0.f )
		{
			CCTAG_THROW( exception::Bug()
				<< exception::dev( "Semi axes must be real positive!" ) );
		}

		_a     = aAux;
		_b     = bAux;
		_angle = thetarad;
	}
}

/*
 * @brief Compute 
 */
void Ellipse::getCanonicForm(Matrix& mCanonic, Matrix& mTprimal, Matrix& mTdual) const 
{

  float q1 = _matrix(0,0);
  float q2 = _matrix(0,1);
  float q3 = _matrix(0,2);
  float q4 = _matrix(1,1);
  float q5 = _matrix(1,2);
  float q6 = _matrix(2,2);
  
  float par1 = q1;
  float par2 = 2*q2;
  float par3 = q4;
  float par4 = 2*q3;
  float par5 = 2*q5;
  
  float thetarad    = 0.5f*atan2(par2,par1 - par3);
  float cost        = cos(thetarad);
  float sint        = sin(thetarad);
  float sin_squared = sint * sint;
  float cos_squared = cost * cost;
  float cos_sin     = sint * cost;

  float Au          = par4 * cost + par5 * sint;
  float Av          = -par4 * sint + par5 * cost;
  float Auu         = par1 * cos_squared + par3 * sin_squared + par2 * cos_sin;
  float Avv         = par1 * sin_squared + par3 * cos_squared - par2 * cos_sin;

  float tuCentre    = - Au/(2*Auu);
  float tvCentre    = - Av/(2*Avv);

  float uCentre     = tuCentre * cost - tvCentre * sint;
  float vCentre     = tuCentre * sint + tvCentre * cost;
  
  float qt1 = cost*(cost*q1 + q2*sint) + sint*(cost*q2 + q4*sint);
  float qt2 = cost*(cost*q2 + q4*sint) - sint*(cost*q1 + q2*sint);
  float qt3 = cost*q3 + q5*sint + uCentre*(cost*q1 + q2*sint) + vCentre*(cost*q2 + q4*sint);
  float qt4 = cost*(cost*q4 - q2*sint) - sint*(cost*q2 - q1*sint);
  float qt5 = cost*q5 - q3*sint + uCentre*(cost*q2 - q1*sint) + vCentre*(cost*q4 - q2*sint);
  float qt6 =  q6 + uCentre*(q3 + q1*uCentre + q2*vCentre) + vCentre*(q5 + q2*uCentre + q4*vCentre) + q3*uCentre + q5*vCentre;
  
  mCanonic(0,0) = qt1;    mCanonic(0,1) = qt2;   mCanonic(0,2) = qt3;
  mCanonic(1,0) = qt2;    mCanonic(1,1) = qt4;   mCanonic(1,2) = qt5;
  mCanonic(2,0) = qt3;    mCanonic(2,1) = qt5;   mCanonic(2,2) = qt6;

  mTprimal(0,0) = cost;   mTprimal(0,1) = sint;  mTprimal(0,2) = - cost*uCentre - sint*vCentre;
  mTprimal(1,0) = -sint;  mTprimal(1,1) = cost;  mTprimal(1,2) = sint*uCentre - cost*vCentre;
  mTprimal(2,0) = 0;      mTprimal(2,1) = 0;     mTprimal(2,2) = cost*cost + sint*sint;
  
  mTdual(0,0) = cost;     mTdual(0,1) = -sint;   mTdual(0,2) = uCentre;
  mTdual(1,0) = sint;     mTdual(1,1) = cost;    mTdual(1,2) = vCentre;
  mTdual(2,0) = 0;        mTdual(2,1) = 0;       mTdual(2,2) = 1.f;
  
}

void Ellipse::computeMatrix()
{
  Eigen::Matrix3f tmp;
  tmp( 0, 0 ) = std::cos( _angle ); tmp( 0, 1 ) = -std::sin( _angle ); tmp( 0, 2 ) = _center.x();
  tmp( 1, 0 ) = std::sin( _angle ); tmp( 1, 1 ) =  std::cos( _angle ); tmp( 1, 2 ) = _center.y();
  tmp( 2, 0 ) =             0.f; tmp( 2, 1 ) =              0.f; tmp( 2, 2 ) =        1.f;

  Eigen::Matrix3f tmpInv;
  Eigen::Matrix3f diag; diag.setIdentity();
  diag( 0, 0 ) =  1.f / ( _a * _a );
  diag( 1, 1 ) =  1.f / ( _b * _b );
  diag( 2, 2 ) = -1.f;
  
  bool invertible;
  tmp.computeInverseWithCheck(tmpInv, invertible);

  if( invertible )
  {
          _matrix = diag * tmpInv;
          _matrix = tmpInv.transpose() * _matrix;
  }
  else
  {
          CCTAG_THROW( exception::Bug()
                          << exception::dev( "Singular matrix!" ) );
  }
}

void scale(const Ellipse & ellipse, Ellipse & rescaleEllipse, float scale)
{
  rescaleEllipse.setCenter(Point2d<Eigen::Vector3f>( ellipse.center().x() * scale, ellipse.center().y() * scale ));
  rescaleEllipse.setA(ellipse.a() * scale);
  rescaleEllipse.setB(ellipse.b() * scale);
  rescaleEllipse.setAngle(ellipse.angle());
}

std::ostream& operator<<(std::ostream& os, const Ellipse& e)
{
  os  << "e = [ " << e.matrix()(0,0) << " " << e.matrix()(0,1) << " " << e.matrix()(0,2) << " ; "
      << e.matrix()(1,0) << " " << e.matrix()(1,1) << " " << e.matrix()(1,2) << " ; "
      << e.matrix()(2,0) << " " << e.matrix()(2,1) << " " << e.matrix()(2,2) << " ] ";
  return os;
}

/* 
 * @brief Sort a set of points by angle along an elliptical arc. Possibly return a subset of these 
 *        points if requested.
 */
void getSortedOuterPoints(
        const Ellipse & ellipse,
        const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & points,
        std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & resPoints,
        std::size_t requestedSize)
{
  // map with the key = angle and the point index
  // Sort points in points by angle
  //std::map<float, std::size_t> mapAngle;
  std::vector< std::pair<float, std::size_t> > vAngles;
  vAngles.reserve(points.size());
  for(std::size_t iPoint = 0 ; iPoint < points.size() ; ++iPoint)
  {
    float angle = std::atan2( points[iPoint].y()- ellipse.center().y() , points[iPoint].x() - ellipse.center().x() );
    vAngles.emplace_back(angle, iPoint);
  }
  
  std::sort (vAngles.begin(), vAngles.end());
  
  // Get the final expected size of resPoints
  const std::size_t nOuterPoints = std::min( requestedSize, points.size() );
  const float step = std::max(1.f,(float) points.size() / (float) ( nOuterPoints - 1 ));
  
  resPoints.clear();
  resPoints.reserve(nOuterPoints);
  
  // Get the final expected size of resPoints
  
  for(std::size_t k = 0 ; ; ++k)
  {
    const std::size_t iToAdd = std::size_t(k*step);
    if ( iToAdd < vAngles.size() )
    {
      resPoints.push_back(points[vAngles[iToAdd].second]);
    }else
    {
      break;
    }
  }
}

}
}
}
