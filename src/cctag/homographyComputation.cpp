#include "CCTag.hpp"
#include "homographyComputation.hpp"

#include <cctag/algebra/lapack.hpp>
#include <cctag/algebra/matrix/operation.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/algebra/determinant.hpp>
#include <cctag/algebra/eig.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>
#include <cctag/progBase/exceptions.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>

#include <algorithm>
#include <cmath>
#include <vector>


namespace rom {
namespace vision {
namespace marker {

void conditionnate( std::pair<rom::vision::marker::CCTag, rom::vision::marker::CCTag> & frameMarkers, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & mT, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & mInvT )
{
	frameMarkers.first.condition(mT, mInvT );
	frameMarkers.second.condition(mT, mInvT );
}

void get2CoplanarCircleConstraint( const boost::numeric::ublas::bounded_matrix<double, 3, 3> & e1,
	                               const boost::numeric::ublas::bounded_matrix<double, 3, 3> & e2,
	                               boost::numeric::ublas::matrix<double> & M )
{
	namespace ublas = boost::numeric::ublas;
	using namespace ublas;
	using namespace rom::numerical;

	bounded_matrix<double, 3, 3> v;
	diagonal_matrix<double> d;
	bounded_matrix<double, 2, 3> S;
	eig( e1, e2, v, d );

	matrix<double> degQuad[3];
	ublas::vector<double> gVector[3];
	for( std::size_t s = 0; s < 3; ++s )
	{
		bounded_matrix<double, 3, 3> X = e1 - d(s, s) * e2;

		std::size_t s1, s2;
		signature( X / normFro( X ), s1, s2 );
		S( 0, s ) = s1;
		S( 1, s ) = s2;

		bounded_matrix<double, 3, 3> uu, vv, ss;
		svds( X, uu, vv, ss, s1 + s2 );

		matrix<double> q = prod( prod< matrix<double> > ( uu, ss ), trans( vv ) );
		q = q / normFro( q );
		degQuad[s] = q / normFro( q );
		gVector[s] = normalize( bounded_vector<double, 3>( column( null( degQuad[s] ), 0 ) ) );
	}

	int nok1 = 0, nok2 = 0;
	std::size_t i = 0;
	double s11v = 0;
	bounded_vector<double, 2> s20v;
	int gvec[2];
	while( ( nok1 != 1 || nok2 != 2 ) && i < 3 )
	{
		if( S(0, i) == 1 && S(1, i) == 1 )
		{
			s11v = d( i, i );
			++nok1;
		}
		if( S(0, i) == 2 && S(1, i) == 0 )
		{
			s20v( nok2 ) = d( i, i );
			gvec[nok2] = i;
			++nok2;
		}
		++i;
	}

	if ( nok1 != 1 || nok2 != 2 )
	{
		ROM_THROW( exception::Bug()
				<< exception::dev("get2CoplanarCircleConstraint failed!") );
	}

	bounded_matrix<double, 3, 2> LD = linesFromRank2Conic( e1 - s11v * e2  );
	bounded_vector<double, 3> linf, radx;

	if ( inner_prod( column( LD, 0 ), gVector[ gvec[ 0 ] ] ) * inner_prod( column( LD, 0 ), gVector[ gvec[ 1 ] ] ) >= 0 &&
		 inner_prod( column( LD, 1 ), gVector[ gvec[ 0 ] ] ) * inner_prod( column( LD, 1 ), gVector[ gvec[ 1 ] ] ) < 0 )
	{
		linf = column( LD, 0 );
		radx = column( LD, 1 );
	}
	else if ( inner_prod( column( LD, 0 ), gVector[ gvec[ 0 ] ] ) * inner_prod( column( LD, 0 ), gVector[ gvec[ 1 ] ] ) < 0 &&
		      inner_prod( column( LD, 1 ), gVector[ gvec[ 0 ] ] ) * inner_prod( column( LD, 1 ), gVector[ gvec[ 1 ] ] ) >= 0 )
	{
		radx = column( LD, 0 );
		linf = column( LD, 1 );
	}
	else
	{
		ROM_THROW( exception::Bug()
				<< exception::dev("get2CoplanarCircleConstraint failed!") );
	}

	M.resize( 7, 6 );
	M.clear();
	M( 0, 0 ) = linf( 0 ); M( 0, 1 ) = linf( 1 ); M( 0, 2 ) = linf( 2 ); M( 0, 3 ) = 0; M( 0, 4 ) = 0; M( 0, 5 ) = 0;
	row( M, 0 ) /= norm_2( row( M, 0 ) );
	M( 1, 0 ) = 0; M( 1, 1 ) = linf(0); M( 1, 2 ) = 0; M( 1, 3 ) = linf(1); M( 1, 4 ) = linf(2); M( 1, 5 ) = 0;
	row( M, 1 ) /= norm_2( row( M, 1 ) );
	M( 2, 0 ) = 0; M( 2, 1 ) = 0; M( 2, 2 ) = linf( 0 ); M( 2, 3 ) = 0; M( 2, 4 ) = linf( 1 ); M( 2, 5 ) = linf( 2 );
	row( M, 2 ) /= norm_2( row( M, 2 ) );

	for( std::size_t s = 0; s < 2; ++s )
	{
		bounded_matrix<double, 3, 2> L;
		bounded_matrix<double, 3, 2> R;
		diagonal_matrix<double> S( 2 );
		svds( degQuad[ gvec[ s ] ], L, R, S, 2 );
		matrix<double> di = prod( L, matSqrt( S ) );
		bounded_vector<double, 6> b1, b2;
		b1(0) = di(0, 1) * di(0, 0);
		b1(1) = di(0, 1) * di(1, 0) + di(1, 1) * di(0, 0);
		b1(2) = di(0, 1) * di(2, 0) + di(2, 1) * di(0, 0);
		b1(3) = di(1, 1) * di(1, 0);
		b1(4) = di(1, 1) * di(2, 0) + di(2, 1) * di(1, 0);
		b1(5) = di(2, 1) * di(2, 0);

		b2(0) = di(0, 0)*di(0, 0) - di(0, 1)*di(0, 1);
		b2(1) = 2.0 * di(0, 0) * di(1, 0) - 2.0 * di(0, 1) * di(1, 1);
		b2(2) = -2.0 * di(0, 1) * di(2, 1) + 2.0 * di(0, 0) * di(2, 0);
		b2(3) = -di(1, 1) * di(1, 1) + di(1, 0) * di(1, 0);
		b2(4) = 2.0 * di(1, 0) * di(2, 0) - 2.0 * di(1, 1) * di(2, 1);
		b2(5) = di(2, 0) * di(2, 0) - di(2, 1) * di(2, 1);

		row( M, 3 + 2 * s ) = b1 / norm_2( b1 );
		row( M, 3 + 2 * s + 1 ) = b2 / norm_2( b2 );
	}
}

void rectifyHomography2PlanarCC(const rom::numerical::geometry::Ellipse & e11, const rom::numerical::geometry::Ellipse & e21, boost::numeric::ublas::bounded_matrix<double, 3, 3> & mH, const double distMarkers)
{
	using namespace boost::numeric::ublas;
	using namespace rom::numerical::geometry;

	Ellipse mQ11 = e11.transform(mH);
	Ellipse mQ21 = e21.transform(mH);

	double a = rom::numerical::distancePoints2D(mQ11.center(),mQ21.center()) / ( 6.0 * distMarkers ); // distance entre les 2 marqueurs = 1 dans le repère monde
	double rho = boost::math::constants::pi<double>()/2 + mQ11.angle();

	bounded_matrix<double, 3, 3> mT;
	double c = cos(rho);
	double s = sin(rho);
	mT(0,0) = c*a; mT(0,1) = -s*a; mT(0,2) = mQ11.center().x();
	mT(1,0) = s*a; mT(1,1) = c*a; mT(1,2) = mQ11.center().y();
	mT(2,0) = 0  ; mT(2,1) = 0  ; mT(2,2) = 1;

	mH = prec_prod(mH,mT);

	mQ21 = e21.transform(mH);

	double x0 = mQ21.center().x();
	double y0 = mQ21.center().y();

	double normD = std::sqrt(x0*x0+y0*y0);

	x0 = x0/normD;
	y0 = y0/normD;

	mT(0,0) = x0; mT(0,1) = -y0; mT(0,2) = 0;
	mT(1,0) = y0; mT(1,1) =  x0; mT(1,2) = 0;
	mT(2,0) = 0 ; mT(2,1) = 0  ; mT(2,2) = 1;

	mH = prec_prod(mH,mT);

	if ( disambiguate( e11, mH ) )
	{
		mT = identity_matrix<double>( 3 );
		mT( 1, 1 ) = -1.0;
		mH = prec_prod( mH, mT );
	}
}


bool disambiguate(const rom::numerical::geometry::Ellipse & q, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & H)
{
	using namespace boost::numeric::ublas;
	using boost::math::constants::pi;
	using namespace rom::numerical;
	//Disambiguate
	Point2dH<double> imP1 = extractEllipsePointAtAngle( q, 0 );
	Point2dH<double> imP2 = extractEllipsePointAtAngle( q, pi<double>() / 4.0 );
	Point2dH<double> imP3 = extractEllipsePointAtAngle( q, pi<double>() / 2.0 );

	bounded_matrix<double, 3, 3> m3Pt;
	column(m3Pt, 0) = imP1;
	column(m3Pt, 1) = imP2;
	column(m3Pt, 2) = imP3;

	double detIm3Pt = rom::numerical::det(m3Pt);

	bounded_matrix<double, 3, 3> invH;
	rom::numerical::invert(H,invH);

	bounded_vector<double,3> p1( prec_prod(invH,imP1) );
	bounded_vector<double,3> p2( prec_prod(invH,imP2) );
	bounded_vector<double,3> p3( prec_prod(invH,imP3) );

	column(m3Pt, 0) = p1;
	column(m3Pt, 1) = p2;
	column(m3Pt, 2) = p3;

	double det3Pt = rom::numerical::det(m3Pt);

	return (detIm3Pt*det3Pt < 0);
}

// Warning : datas in cctags have to be conditionnate (vgg_conditioner_from_pts or vgg_conditioner_from_image (mainly because signature run in same conditions).

void homographyFrom2CPlanar( const std::pair<CCTag, CCTag> & cctags, boost::numeric::ublas::bounded_matrix<double, 3, 3> & h )
{
	namespace ublas = boost::numeric::ublas;
	using namespace ublas;

	using namespace rom::numerical::geometry;

	matrix<double> MM( 21, 6 );
	matrix<double> M[3];

	std::vector< rom::numerical::geometry::Ellipse > ellipses1 = cctags.first.ellipses();
	std::vector< rom::numerical::geometry::Ellipse > ellipses2 = cctags.second.ellipses();
	std::size_t sz1 = ellipses1.size();
	std::size_t sz2 = ellipses2.size();
	assert( sz1 >= 3 );
	assert( sz2 >= 3 );
	MM.clear();

	Ellipse e11 = ellipses1[sz1 - 1];
	Ellipse e12 = ellipses1[1];
	Ellipse e21 = ellipses2[sz1 - 1];
	Ellipse e22 = ellipses2[1];

	get2CoplanarCircleConstraint( e11.matrix(), e21.matrix(), M[0] );
	std::size_t k = 0;
	project( MM, range( k, M[0].size1() ), range( 0, M[0].size2() ) ) = M[0];
	get2CoplanarCircleConstraint( e11.matrix(), e22.matrix(), M[1] );
	k += M[0].size1();
	project( MM, range( k, k + M[1].size1() ), range( 0, M[1].size2() ) ) = M[1];
	get2CoplanarCircleConstraint( e12.matrix(), e21.matrix(), M[2] );
	k += M[1].size1();
	project( MM, range( k, k + M[2].size1() ), range( 0, M[2].size2() ) ) = M[2];

	matrix<double> U, V;
	diagonal_matrix<double> S;
	rom::numerical::svd( MM, U, V, S );
	ublas::vector<double> xTLS = column( V, MM.size2() - 1 );
	bounded_matrix<double, 3, 3> cdcp;
	row( cdcp, 0 ) = subrange( xTLS, 0, 3 );
	cdcp( 1, 0 ) = xTLS( 1 );

	///@todo: I fixed the following line but I am not sure if it is correct. Check this.
	///old: matrix_vector_slice< bounded_matrix<double, 3, 3> > ( cdcp, slice(1, 0, 2), slice(1, 1, 3) ) = subrange( xTLS, 3, 5 );
	matrix_vector_slice< bounded_matrix<double, 3, 3> > ( cdcp, slice(1, 0, 2), slice(1, 1, 2) ) = subrange( xTLS, 3, 5 );
	cdcp( 2, 0 ) = xTLS( 2 );
	cdcp( 2, 1 ) = xTLS( 4 );
	cdcp( 2, 2 ) = xTLS( 5 );

	rom::numerical::svd( cdcp, U, V, S );
	assert( S.size1() > 2 );
	S( 0, 0 ) = std::sqrt( S( 0, 0 ) );
	S( 1, 1 ) = std::sqrt( S( 1, 1 ) );
	S( 2, 2 ) = 1.0;

	h = prec_prod(U,S);
}

}
}
}
