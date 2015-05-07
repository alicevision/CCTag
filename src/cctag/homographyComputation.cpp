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


namespace cctag {
namespace vision {
namespace marker {

void conditionnate( std::pair<cctag::vision::marker::CCTag, cctag::vision::marker::CCTag> & frameMarkers, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & mT, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & mInvT )
{
	frameMarkers.first.condition(mT, mInvT );
	frameMarkers.second.condition(mT, mInvT );
}

void get2CoplanarCircleConstraint( const boost::numeric::ublas::bounded_matrix<double, 3, 3> & e1,
	                               const boost::numeric::ublas::bounded_matrix<double, 3, 3> & e2,
	                               boost::numeric::ublas::matrix<double> & M )
{
	namespace ublas = boost::numeric::ublas;

	ublas::bounded_matrix<double, 3, 3> v;
	ublas::diagonal_matrix<double> d;
	ublas::bounded_matrix<double, 2, 3> S;
	cctag::numerical::eig( e1, e2, v, d );

	ublas::matrix<double> degQuad[3];
	ublas::vector<double> gVector[3];
	for( std::size_t s = 0; s < 3; ++s )
	{
		ublas::bounded_matrix<double, 3, 3> X = e1 - d(s, s) * e2;

		std::size_t s1, s2;
		signature( X / cctag::numerical::normFro( X ), s1, s2 );
		S( 0, s ) = s1;
		S( 1, s ) = s2;

		ublas::bounded_matrix<double, 3, 3> uu, vv, ss;
		cctag::numerical::svds( X, uu, vv, ss, s1 + s2 );

		ublas::matrix<double> q = ublas::prod( ublas::prod< ublas::matrix<double> > ( uu, ss ), ublas::trans( vv ) );
		q = q / cctag::numerical::normFro( q );
		degQuad[s] = q / cctag::numerical::normFro( q );
		gVector[s] = cctag::numerical::normalize( ublas::bounded_vector<double, 3>( ublas::column( null( degQuad[s] ), 0 ) ) );
	}
        
        int nok1 = 0, nok2 = 0;
        std::size_t i = 0;
        double s11v = 0;
        ublas::bounded_vector<double, 2> s20v;
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
            CCTAG_THROW( exception::Bug()
            << exception::dev("get2CoplanarCircleConstraint failed!") );
        }
        ublas::bounded_matrix<double, 3, 2> LD = linesFromRank2Conic( e1 - s11v * e2 );
        ublas::bounded_vector<double, 3> linf, radx;
        if ( ublas::inner_prod( ublas::column( LD, 0 ), gVector[ gvec[ 0 ] ] ) * ublas::inner_prod( ublas::column( LD, 0 ), gVector[ gvec[ 1 ] ] ) >= 0 &&
            ublas::inner_prod( ublas::column( LD, 1 ), gVector[ gvec[ 0 ] ] ) * ublas::inner_prod( ublas::column( LD, 1 ), gVector[ gvec[ 1 ] ] ) < 0 )
        {
            linf = ublas::column( LD, 0 );
            radx = ublas::column( LD, 1 );
        }
        else if ( ublas::inner_prod( ublas::column( LD, 0 ), gVector[ gvec[ 0 ] ] ) * ublas::inner_prod( ublas::column( LD, 0 ), gVector[ gvec[ 1 ] ] ) < 0 &&
            ublas::inner_prod( ublas::column( LD, 1 ), gVector[ gvec[ 0 ] ] ) * ublas::inner_prod( ublas::column( LD, 1 ), gVector[ gvec[ 1 ] ] ) >= 0 )
        {
            radx = ublas::column( LD, 0 );
            linf = ublas::column( LD, 1 );
        }
        else
        {
            CCTAG_THROW( exception::Bug()
            << exception::dev("get2CoplanarCircleConstraint failed!") );
        }
        
        
        M.resize( 7, 6 );
        M.clear();
        M( 0, 0 ) = linf( 0 ); M( 0, 1 ) = linf( 1 ); M( 0, 2 ) = linf( 2 ); M( 0, 3 ) = 0; M( 0, 4 ) = 0; M( 0, 5 ) = 0;
        row( M, 0 ) /= ublas::norm_2( ublas::row( M, 0 ) );
        M( 1, 0 ) = 0; M( 1, 1 ) = linf(0); M( 1, 2 ) = 0; M( 1, 3 ) = linf(1); M( 1, 4 ) = linf(2); M( 1, 5 ) = 0;
        row( M, 1 ) /= ublas::norm_2( ublas::row( M, 1 ) );
        M( 2, 0 ) = 0; M( 2, 1 ) = 0; M( 2, 2 ) = linf( 0 ); M( 2, 3 ) = 0; M( 2, 4 ) = linf( 1 ); M( 2, 5 ) = linf( 2 );
        row( M, 2 ) /= ublas::norm_2( ublas::row( M, 2 ) );
        for( std::size_t s = 0; s < 2; ++s )
        {
            ublas::bounded_matrix<double, 3, 2> L;
            ublas::bounded_matrix<double, 3, 2> R;
            ublas::diagonal_matrix<double> S( 2 );
            cctag::numerical::svds( degQuad[ gvec[ s ] ], L, R, S, 2 );
            ublas::matrix<double> di = ublas::prod( L, cctag::numerical::matSqrt( S ) );
            ublas::bounded_vector<double, 6> b1, b2;
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
            ublas::row( M, 3 + 2 * s ) = b1 / norm_2( b1 );
            ublas::row( M, 3 + 2 * s + 1 ) = b2 / ublas::norm_2( b2 );
        }
}

void rectifyHomography2PlanarCC(const cctag::numerical::geometry::Ellipse & e11, const cctag::numerical::geometry::Ellipse & e21, boost::numeric::ublas::bounded_matrix<double, 3, 3> & mH, const double distMarkers)
{
	namespace ublas = boost::numeric::ublas;
	using namespace cctag::numerical::geometry;

	Ellipse mQ11 = e11.transform(mH);
	Ellipse mQ21 = e21.transform(mH);

	double a = cctag::numerical::distancePoints2D(mQ11.center(),mQ21.center()) / ( 6.0 * distMarkers ); // distance entre les 2 marqueurs = 1 dans le repère monde
	double rho = boost::math::constants::pi<double>()/2 + mQ11.angle();

	ublas::bounded_matrix<double, 3, 3> mT;
	double c = cos(rho);
	double s = sin(rho);
	mT(0,0) = c*a; mT(0,1) = -s*a; mT(0,2) = mQ11.center().x();
	mT(1,0) = s*a; mT(1,1) = c*a; mT(1,2) = mQ11.center().y();
	mT(2,0) = 0  ; mT(2,1) = 0  ; mT(2,2) = 1;

	mH = ublas::prec_prod(mH,mT);

	mQ21 = e21.transform(mH);

	double x0 = mQ21.center().x();
	double y0 = mQ21.center().y();

	double normD = std::sqrt(x0*x0+y0*y0);

	x0 = x0/normD;
	y0 = y0/normD;

	mT(0,0) = x0; mT(0,1) = -y0; mT(0,2) = 0;
	mT(1,0) = y0; mT(1,1) =  x0; mT(1,2) = 0;
	mT(2,0) = 0 ; mT(2,1) = 0  ; mT(2,2) = 1;

	mH = ublas::prec_prod(mH,mT);

	if ( disambiguate( e11, mH ) )
	{
		mT = ublas::identity_matrix<double>( 3 );
		mT( 1, 1 ) = -1.0;
		mH = ublas::prec_prod( mH, mT );
	}
}


bool disambiguate(const cctag::numerical::geometry::Ellipse & q, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & H)
{
	namespace ublas = boost::numeric::ublas;
	using boost::math::constants::pi;
	using namespace cctag::numerical;
	//Disambiguate
	Point2dH<double> imP1 = extractEllipsePointAtAngle( q, 0 );
	Point2dH<double> imP2 = extractEllipsePointAtAngle( q, pi<double>() / 4.0 );
	Point2dH<double> imP3 = extractEllipsePointAtAngle( q, pi<double>() / 2.0 );

	ublas::bounded_matrix<double, 3, 3> m3Pt;
	ublas::column(m3Pt, 0) = imP1;
	ublas::column(m3Pt, 1) = imP2;
	ublas::column(m3Pt, 2) = imP3;

	double detIm3Pt = cctag::numerical::det(m3Pt);

	ublas::bounded_matrix<double, 3, 3> invH;
	cctag::numerical::invert(H,invH);

	ublas::bounded_vector<double,3> p1( ublas::prec_prod(invH,imP1) );
	ublas::bounded_vector<double,3> p2( ublas::prec_prod(invH,imP2) );
	ublas::bounded_vector<double,3> p3( ublas::prec_prod(invH,imP3) );

	ublas::column(m3Pt, 0) = p1;
	ublas::column(m3Pt, 1) = p2;
	ublas::column(m3Pt, 2) = p3;

	double det3Pt = cctag::numerical::det(m3Pt);

	return (detIm3Pt*det3Pt < 0);
}

// Warning : datas in cctags have to be conditionnate (vgg_conditioner_from_pts or vgg_conditioner_from_image (mainly because signature run in same conditions).

void homographyFrom2CPlanar( const std::pair<CCTag, CCTag> & cctags, boost::numeric::ublas::bounded_matrix<double, 3, 3> & h )
{
	namespace ublas = boost::numeric::ublas;
	using namespace cctag::numerical::geometry;

	ublas::matrix<double> MM( 21, 6 );
	ublas::matrix<double> M[3];

	std::vector< cctag::numerical::geometry::Ellipse > ellipses1 = cctags.first.ellipses();
	std::vector< cctag::numerical::geometry::Ellipse > ellipses2 = cctags.second.ellipses();
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
	project( MM, ublas::range( k, M[0].size1() ), ublas::range( 0, M[0].size2() ) ) = M[0];
	get2CoplanarCircleConstraint( e11.matrix(), e22.matrix(), M[1] );
	k += M[0].size1();
	project( MM, ublas::range( k, k + M[1].size1() ), ublas::range( 0, M[1].size2() ) ) = M[1];
	get2CoplanarCircleConstraint( e12.matrix(), e21.matrix(), M[2] );
	k += M[1].size1();
	project( MM, ublas::range( k, k + M[2].size1() ), ublas::range( 0, M[2].size2() ) ) = M[2];

	ublas::matrix<double> U, V;
	ublas::diagonal_matrix<double> S;
	cctag::numerical::svd( MM, U, V, S );
	ublas::vector<double> xTLS = ublas::column( V, MM.size2() - 1 );
	ublas::bounded_matrix<double, 3, 3> cdcp;
	ublas::row( cdcp, 0 ) = ublas::subrange( xTLS, 0, 3 );
	cdcp( 1, 0 ) = xTLS( 1 );

	///@todo: I fixed the following line but I am not sure if it is correct. Check this.
	///old: matrix_vector_slice< bounded_matrix<double, 3, 3> > ( cdcp, slice(1, 0, 2), slice(1, 1, 3) ) = subrange( xTLS, 3, 5 );
	ublas::matrix_vector_slice< ublas::bounded_matrix<double, 3, 3> > ( cdcp, ublas::slice(1, 0, 2), ublas::slice(1, 1, 2) ) = ublas::subrange( xTLS, 3, 5 );
	cdcp( 2, 0 ) = xTLS( 2 );
	cdcp( 2, 1 ) = xTLS( 4 );
	cdcp( 2, 2 ) = xTLS( 5 );

	cctag::numerical::svd( cdcp, U, V, S );
	assert( S.size1() > 2 );
	S( 0, 0 ) = std::sqrt( S( 0, 0 ) );
	S( 1, 1 ) = std::sqrt( S( 1, 1 ) );
	S( 2, 2 ) = 1.0;

	h = ublas::prec_prod(U,S);
}

}
}
}
