/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <math.h>
#include "geom_ellipse.h"
#include "math_constants.h" // a CUDA header file
#include "debug_macros.hpp"

using namespace std;

namespace cctag {
namespace geometry {


__host__ __device__
ellipse::ellipse( const matrix3x3& matrix )
	: _matrix( matrix )
{
	computeParameters();
}

__host__ __device__
ellipse::ellipse( const float2& center, const float a, const float b, const float angle )
{
	setParameters( center, a, b, angle );
}

__host__ __device__
void ellipse::setMatrix( const matrix3x3& matrix )
{
	_matrix = matrix;
	computeParameters();
}

__host__ __device__
void ellipse::setParameters( const float2& center, const float a, const float b, const float angle )
{
	if( a < 0.0 || b < 0.0 )
	{
        crash( __FILE__, __LINE__, "Semi axes must be real positive!" );
        return;
	}
	_center = center;
	_a      = a;
	_b      = b;
    _angle  = angle;
	computeMatrix();
}

__host__ __device__
void ellipse::setCenter( const float2& center )
{
	_center = center;
	computeMatrix();
}

__host__ __device__
void ellipse::setA( const float a )
{
	if( a < 0.0 )
	{
        crash( __FILE__, __LINE__, "Semi axes must be real positive!" );
        return;
	}
	_a = a;
	computeMatrix();
}

__host__ __device__
void ellipse::setB( const float b )
{
	if( b < 0.0 )
	{
        crash( __FILE__, __LINE__, "Semi axes must be real positive!" );
        return;
	}
	_b = b;
	computeMatrix();
}

__host__ __device__
void ellipse::setAngle( const float angle )
{
	_angle = angle;
	computeMatrix();
}

#if 0
__host__ __device__
ellipse ellipse::transform( const matrix3x3& mT ) const
{
    matrix3x3_tView mtransposed( mT ); // not a copy!
	const matrix3x3 a   = cctag::geometry::prod( mtransposed, _matrix );
	const matrix3x3 mET = cctag::geometry::prod( a, mT );
	return ellipse( mET );
}
#endif

__host__ __device__
void ellipse::projectiveTransform( const matrix3x3& transf_m, ellipse& e ) const
{
    matrix3x3_tView transf_m_transposed( transf_m );
    e.setMatrix(
        prod( transf_m_transposed,
              prod( _matrix,
                    transf_m ) ) );
}

__host__ __device__
void ellipse::computeParameters()
{
#ifdef __CUDA_ARCH__
    float par[6];
	par[0] = _matrix( 0, 0 );
	par[1] = ldexpf( _matrix( 0, 1 ), 1 ); 2.0f * _matrix( 0, 1 );
	par[2] = _matrix( 1, 1 );
	par[3] = ldexpf( _matrix( 0, 2 ), 1 ); 2.0f * _matrix( 0, 2 );
	par[4] = ldexpf( _matrix( 1, 2 ), 1 ); 2.0f * _matrix( 1, 2 );
	par[5] = _matrix( 2, 2 );

	// const float thetarad    = 0.5 * std::atan2( par[1], par[0] - par[2] );
	const float thetarad    = 0.5 * atan2f( par[1], par[0] - par[2] );
	float cost; //        = std::cos( thetarad );
	float sint; //        = std::sin( thetarad );
    __sincosf( thetarad, &sint, &cost );
	const float sin_squared = sint * sint;
	const float cos_squared = cost * cost;
	const float cos_sin     = sint * cost;

	const float Ao  =  par[5];
	const float Au  =  par[3] * cost + par[4] * sint;
	const float Av  = -par[3] * sint + par[4] * cost;
	const float Auu =  par[0] * cos_squared + par[2] * sin_squared + par[1] * cos_sin;
	const float Avv =  par[0] * sin_squared + par[2] * cos_squared - par[1] * cos_sin;

	if( Auu == 0 || Avv == 0 )
	{
		_center = make_float2( 0.0f, 0.0f );
		_a      = 0.0f;
		_b      = 0.0f;
		_angle  = 0.0f;
	}
	else
	{
        const float tuCentre = -Au * __frcp_rn( ldexpf( Auu, 1 ) ); // -Au/(2.0f*Auu);
        const float tvCentre = -Av * __frcp_rn( ldexpf( Avv, 1 ) ); // -Av/(2.0f*Avv);
		const float wCentre  = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;

		_center = make_float2( tuCentre * cost - tvCentre * sint,
                               tuCentre * sint + tvCentre * cost );

		const float Ru = -wCentre * __frcp_rn( Auu ); // -wCentre / Auu;
		const float Rv = -wCentre * __frcp_rn( Avv ); // -wCentre / Avv;

		// const float aAux = std::sqrt( std::abs( Ru ) ) * boost::math::sign( Ru );
		// const float bAux = std::sqrt( std::abs( Rv ) ) * boost::math::sign( Rv );
		const float aAux = copysignf( __fsqrt_rn( fabsf( Ru ) ), Ru );
		const float bAux = copysignf( __fsqrt_rn( fabsf( Rv ) ), Rv );

		if( aAux < 0.0f || bAux < 0.0f ) {
            crash( __FILE__, __LINE__, "Semi axes must be real positive!" );
            return;
		}

		_a     = aAux;
		_b     = bAux;
		_angle = thetarad;
	}
#else // not __CUDA_ARCH__
    float par[6];
	par[0] = _matrix( 0, 0 );
	par[1] = ldexpf( _matrix( 0, 1 ), 1 ); 2.0f * _matrix( 0, 1 );
	par[2] = _matrix( 1, 1 );
	par[3] = ldexpf( _matrix( 0, 2 ), 1 ); 2.0f * _matrix( 0, 2 );
	par[4] = ldexpf( _matrix( 1, 2 ), 1 ); 2.0f * _matrix( 1, 2 );
	par[5] = _matrix( 2, 2 );

	const float thetarad    = 0.5f * std::atan2( par[1], par[0] - par[2] );
	const float cost = std::cos( thetarad );
	const float sint = std::sin( thetarad );
	const float sin_squared = sint * sint;
	const float cos_squared = cost * cost;
	const float cos_sin     = sint * cost;

	const float Ao  =  par[5];
	const float Au  =  par[3] * cost + par[4] * sint;
	const float Av  = -par[3] * sint + par[4] * cost;
	const float Auu =  par[0] * cos_squared + par[2] * sin_squared + par[1] * cos_sin;
	const float Avv =  par[0] * sin_squared + par[2] * cos_squared - par[1] * cos_sin;

	if( Auu == 0 || Avv == 0 )
	{
		_center = make_float2( 0.0f, 0.0f );
		_a      = 0.0f;
		_b      = 0.0f;
		_angle  = 0.0f;
	}
	else
	{
        const float tuCentre = -Au/(2.0f*Auu);
        const float tvCentre = -Av/(2.0f*Avv);
		const float wCentre  = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;

		_center = make_float2( tuCentre * cost - tvCentre * sint,
                               tuCentre * sint + tvCentre * cost );

		const float Ru = -wCentre / Auu;
		const float Rv = -wCentre / Avv;

		// const float aAux = std::sqrt( std::abs( Ru ) ) * boost::math::sign( Ru );
		// const float bAux = std::sqrt( std::abs( Rv ) ) * boost::math::sign( Rv );
		const float aAux = ::copysignf( std::sqrt( std::abs( Ru ) ), Ru );
		const float bAux = ::copysignf( std::sqrt( std::abs( Rv ) ), Rv );

		if( aAux < 0.0f || bAux < 0.0f ) {
            crash( __FILE__, __LINE__, "Semi axes must be real positive!" );
            return;
		}

		_a     = aAux;
		_b     = bAux;
		_angle = thetarad;
	}
#endif // not __CUDA_ARCH__
}

/*
 * @brief Compute 
 */
__device__
void ellipse::getCanonicForm(matrix3x3& mCanonic, matrix3x3& mTprimal, matrix3x3& mTdual) const 
{
    float q1 = _matrix(0,0);
    float q2 = _matrix(0,1);
    float q3 = _matrix(0,2);
    float q4 = _matrix(1,1);
    float q5 = _matrix(1,2);
    float q6 = _matrix(2,2);
  
    float par1 = q1;
    float par2 = ldexpf( q2, 1 ); // 2.0f*q2;
    float par3 = q4;
    float par4 = ldexpf( q3, 1 ); // 2.0f*q3;
    float par5 = ldexpf( q5, 1 ); // 2.0f*q5;
    // float par6 = q6;
  
    float thetarad    = 0.5f*atan2f( par2, par1 - par3 );
    float cost; //        = cos(thetarad);
    float sint; //        = sin(thetarad);
    __sincosf( thetarad, &sint, &cost );
    float sin_squared = sint * sint;
    float cos_squared = cost * cost;
    float cos_sin     = sint * cost;

    // float Ao          = par6;
    float Au          = par4 * cost + par5 * sint;
    float Av          = -par4 * sint + par5 * cost;
    float Auu         = par1 * cos_squared + par3 * sin_squared + par2 * cos_sin;
    float Avv         = par1 * sin_squared + par3 * cos_squared - par2 * cos_sin;

    float tuCentre    = - Au * __frcp_rn( ldexpf( Auu, 1 ) ); // - Au/(2.0f*Auu);
    float tvCentre    = - Av * __frcp_rn( ldexpf( Avv, 1 ) ); // - Av/(2.0f*Avv);

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
    mTprimal(2,0) = 0.0f;   mTprimal(2,1) = 0.0f;  mTprimal(2,2) = cost*cost + sint*sint;
  
    mTdual(0,0) = cost;     mTdual(0,1) = -sint;   mTdual(0,2) = uCentre;
    mTdual(1,0) = sint;     mTdual(1,1) = cost;    mTdual(1,2) = vCentre;
    mTdual(2,0) = 0.0f;     mTdual(2,1) = 0.0f;    mTdual(2,2) = 1.0f;
}

__host__ __device__
void ellipse::computeMatrix()
{
#ifdef __CUDA_ARCH__
    matrix3x3 tmp;
    float cosa;
    float sina;
    __sincosf( _angle, &sina, &cosa );
    tmp( 0, 0 ) =  cosa; //  std::cos( _angle );
    tmp( 0, 1 ) = -sina; // -std::sin( _angle );
    tmp( 0, 2 ) =  _center.x;
    tmp( 1, 0 ) =  sina; // std::sin( _angle );
    tmp( 1, 1 ) =  cosa; // std::cos( _angle );
    tmp( 1, 2 ) =  _center.y;
    tmp( 2, 0 ) = 0.0f;
    tmp( 2, 1 ) = 0.0f;
    tmp( 2, 2 ) = 1.0f;

    matrix3x3 tmpInv;
    matrix3x3 diag;
    diag.setDiag( __frcp_rn( _a * _a ), // 1.0f / ( _a * _a ),
                  __frcp_rn( _b * _b ), // 1.0f / ( _b * _b ),
                  -1.0f );

    if( tmp.invert( tmpInv ) )
    {
        _matrix = cctag::geometry::prod( diag, tmpInv );
        matrix3x3_tView tmpInvTrans( tmpInv ); // not a copy!
        _matrix = cctag::geometry::prod( tmpInvTrans, _matrix );
    }
    else
    {
        crash( __FILE__, __LINE__, "Singular matrix!" );
        return;
    }
#else // not __CUDA_ARCH__
    matrix3x3 tmp;
    tmp( 0, 0 ) =  std::cos( _angle );
    tmp( 0, 1 ) = -std::sin( _angle );
    tmp( 0, 2 ) =  _center.x;
    tmp( 1, 0 ) =  std::sin( _angle );
    tmp( 1, 1 ) =  std::cos( _angle );
    tmp( 1, 2 ) =  _center.y;
    tmp( 2, 0 ) = 0.0f;
    tmp( 2, 1 ) = 0.0f;
    tmp( 2, 2 ) = 1.0f;

    matrix3x3 tmpInv;
    matrix3x3 diag;
    diag.setDiag( 1.0f / ( _a * _a ),
                  1.0f / ( _b * _b ),
                  -1.0f );

    if( tmp.invert( tmpInv ) )
    {
        _matrix = cctag::geometry::prod( diag, tmpInv );
        matrix3x3_tView tmpInvTrans( tmpInv ); // not a copy!
        _matrix = cctag::geometry::prod( tmpInvTrans, _matrix );
    }
    else
    {
        crash( __FILE__, __LINE__, "Singular matrix!" );
        return;
    }
#endif // not __CUDA_ARCH__
}

__device__
void ellipse::computeHomographyFromImagedCenter( const float2 center, matrix3x3& mHomography ) const
{
    matrix3x3 mCanonic;
    matrix3x3 mTCan;
    matrix3x3 mTInvCan;

    this->getCanonicForm(mCanonic, mTCan, mTInvCan);

    // Get the center coordinates in the new canonical representation.
    float2 c = mTCan.applyHomography( center );

    // Closed-form solution for the homography plan->image computation
    // The ellipse is supposed to be in its canonical representation

    // Matlab closed-form
    //H =
    //[ [     Q33,        Q22*xc*yc, -Q33*xc];
    //  [       0,      - Q11*xc^2 - Q33, -Q33*yc];
    //  [ -Q11*xc,        Q22*yc,    -Q33] ] ...
    // * diag([ ((Q22*Q33/Q11*(Q11*xc^2 + Q22*yc^2 + Q33)))^(1/2);Q33;(-Q22*(Q11*xc^2 + Q33))^(1/2)]);


    float Q11 = mCanonic(0,0);
    float Q22 = mCanonic(1,1);
    float Q33 = mCanonic(2,2);

    float xc2 = c.x * c.x;
    float yc2 = c.y * c.y;

    mHomography(0,0) = Q33;
    mHomography(1,0) = 0.0;
    mHomography(2,0) = -Q11*c.x;

    mHomography(0,1) = Q22*c.x*c.y;
    mHomography(1,1) = -Q11*xc2-Q33;
    mHomography(2,1) = Q22*c.y;

    mHomography(0,2) = -Q33*c.x;
    mHomography(1,2) = -Q33*c.y;
    mHomography(2,2) = -Q33;

    float mDiag[3];
    mDiag[0] = __fsqrt_rn( (Q22*Q33*__frcp_rn(Q11)*(Q11*xc2 + Q22*yc2 + Q33)) );
    mDiag[1] = Q33;
    mDiag[2] = __fsqrt_rn( -Q22*(Q11*xc2 + Q33) );

    for(int i=0; i < 3 ; ++i)
    {
        for(int j=0; j < 3 ; ++j)
        {
            mHomography(i,j) *= mDiag[j];
        }
    }

    mHomography = cctag::geometry::prod( mTInvCan, mHomography ); // mHomography = mTInvCan*mHomography
}

__host__ __device__
void ellipse::makeConditionerFromEllipse( matrix3x3& output ) const
{
    const float meanAB = ( _a * _b ) / 2.0f;

    output(0,0) = CUDART_SQRT_TWO_F / meanAB;
    output(0,1) = 0.0f;
    output(0,2) = -CUDART_SQRT_TWO_F * _center.x / meanAB;

    output(1,0) = 0.0f;
    output(1,1) = CUDART_SQRT_TWO_F / meanAB;
    output(1,2) = -CUDART_SQRT_TWO_F * _center.y / meanAB;

    output(2,0) = 0.0f;
    output(2,1) = 0.0f;
    output(2,2) = 1.0f;
}


__host__ __device__
void ellipse::crash( const char* file, int line, const char* msg )
{
#ifdef __CUDA_ARCH__
    printf( "block (%d,%d,%d) thread (%d,%d,%d) crashes in %s %d: %s\n",
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z,
            file, line, msg );
    assert(0);
#else // not __CUDA_ARCH__
    POP_FATAL_FL( msg, file, line );
#endif // not __CUDA_ARCH__
}

#if 0
__host__
std::ostream& operator<<(std::ostream& os, const ellipse& e)
{
    os  << "e = [ " << e.matrix()(0,0) << " " << e.matrix()(0,1) << " " << e.matrix()(0,2) << " ; "
        << e.matrix()(1,0) << " " << e.matrix()(1,1) << " " << e.matrix()(1,2) << " ; "
        << e.matrix()(2,0) << " " << e.matrix()(2,1) << " " << e.matrix()(2,2) << " ] ";
  return os;
}
#endif

}; // namespace geometry
}; // namespace cctag

