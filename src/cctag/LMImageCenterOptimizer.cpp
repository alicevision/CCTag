#ifdef WITH_CMINPACK
#include "LMImageCenterOptimizer.hpp"

#include <cctag/optimization/conditioner.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/geometry/distance.hpp>

#include <cminpack.h>

namespace cctag {
namespace vision {
namespace marker {

LMImageCenterOptimizer::LMImageCenterOptimizer()
{
}

LMImageCenterOptimizer::~LMImageCenterOptimizer()
{
}

double LMImageCenterOptimizer::operator()( CCTag & cctagToRefine )
{
	using namespace cctag::numerical;
	// La transformation T ramène tous les points dans le "rectangle unité"
	ublas::bounded_matrix<double, 3, 3> T = optimization::conditionerFromPoints( cctagToRefine.points()[cctagToRefine.points().size() - 1] );
	ublas::bounded_matrix<double, 3, 3> TInv;
	cctag::numerical::invert( T, TInv );

	// Conditionne le marqueur avant la minimisation
	cctagToRefine.condition( T, TInv );

	// Calcul du centre de gravité de points(0)
	ublas::bounded_vector<double, 3> o = cctag::numerical::mean( cctagToRefine.points()[0] );

	//	std::cout << o << std::endl ;

	//	vector<double> x0 = // [ o(1) o(2) ; algo_impl.radiusRatio ] // // algo_impl.radiusRatio est un vecteur de taille
	// numEllipses-1 contenant les rapports des rayons pour
	// L'initialisation de lmdif_

	int m = 0; // nombre d'equations
	std::vector< std::vector< Point2dN<double> > >::const_iterator ite = cctagToRefine.points().end() - 1;
	for( std::vector< std::vector< Point2dN<double> > >::const_iterator it = cctagToRefine.points().begin(); it != ite; ++it )
	{
		m += it->size();
	}

	//	// Initialisation des paramètres de minimisation
	int n    = 2 + cctagToRefine.radiusRatios().size();
	std::vector<int> iwa( n );
	std::vector<double> fvec( m );
	std::vector<double> wa( m * n + 5 * n + m );
	std::vector<double> xInit( n );
	double ftol   = 0.00000001;
	double xtol   = 0.00000001;
	double gtol   = 0.00000001;
	double epsfcn = 0.00000001;
	double factor = 100.f;
	int maxfev    = 300;
	int mode      = 1;
	int nprint    = 1;
	int nfev;
	int ldfjac = m;
	std::vector<int> ipvt( n );
	std::vector<double> diag( n );
	std::vector<double> fjac( m * n );
	std::vector<double> qtf( n );
	std::vector<double> wa1( n );
	std::vector<double> wa2( n );
	std::vector<double> wa3( n );
	std::vector<double> wa4( m );

	//Initialization of initial vector X0
	xInit[0] = o( 0 );
	xInit[1] = o( 1 );
	for( std::size_t i = 0; i < cctagToRefine.radiusRatiosInit().size(); ++i )
	{
		xInit[i + 2] = cctagToRefine.radiusRatiosInit()[i];
	}

	// Minimization
	lmdif( &homology, &cctagToRefine, m, n, &xInit[0], &fvec[0], ftol, xtol, gtol, maxfev, epsfcn, &diag[0], mode, factor, nprint, &nfev, &fjac[0], ldfjac, &ipvt[0], &qtf[0], &wa1[0], &wa2[0], &wa3[0], &wa4[0] );

	double res = 0;

	for( std::size_t i = 0; i < fvec.size(); ++i )
	{
		res += ( fvec[i] ) * ( fvec[i] );
	}

	cctagToRefine.condition( TInv, T );

	return res / fvec.size();
}

int LMImageCenterOptimizer::homology( void* p, int m, int n, const double* x, double* fvec, int iflag )
{
	CCTag* cctag = static_cast<CCTag*>(p);

	using namespace boost::numeric::ublas;

	double f = 1;

	identity_matrix<double> eye( 3, 3 );

	cctag->centerImg().setX( x[0] );
	cctag->centerImg().setY( x[1] );

	const bounded_matrix<double, 3, 3> & Q0   = cctag->outerEllipse().matrix();
	bounded_vector<double, 3> l       = prec_prod( Q0, cctag->centerImg() );
	bounded_matrix<double, 3, 3> GAux = outer_prod( cctag->centerImg(), l ) / inner_prod( cctag->centerImg(), l );

	int i = 0;
	int j = 0;

	std::vector< std::vector< Point2dN<double> > >::const_iterator itEllipsesEnd = cctag->points().end() - 1;
	const double* currentX = &x[2];
	std::vector<double>::iterator itRadius = cctag->radiusRatios().begin();

	for( std::vector< std::vector< Point2dN<double> > >::const_iterator itEllipsesPts = cctag->points().begin(); itEllipsesPts != itEllipsesEnd; ++itEllipsesPts )
	{
		*itRadius = *currentX;

		bounded_matrix<double, 3, 3> GInv = eye + ( 1.0 / *currentX - 1.0 ) * GAux;

		bounded_matrix<double, 3, 3> Q1 = prec_prod( bounded_matrix<double, 3, 3>( trans( GInv ) ), bounded_matrix<double, 3, 3>( prec_prod( Q0, GInv ) ) );

		cctag::numerical::geometry::Ellipse l;
		l.setMatrix( Q1 );

		BOOST_FOREACH( const Point2dN<double> &p, *itEllipsesPts )
		{
			fvec[j] = cctag::numerical::distancePointEllipse( p, l, f );
			++j;
		}
		++i;
		++itRadius;
		++currentX;
	}

	if( ( x[2] <= 1 ) || ( x[2] > 6 ) || std::isnan( x[2] ) )
	{
		iflag = -1;
		return iflag;
	}
	currentX = &x[2];
	const double* currentXp1 = currentX + 1;
	const double* endX = &x[n-1];
	while( currentXp1 < endX )
	{
		if( ( *currentXp1 - *currentX > 0 ) || ( *currentXp1 <= 1 ) || ( *currentXp1 > 6 ) || std::isnan( *currentXp1 ) )
		{
			iflag = -1;
			return iflag;
		}
		++currentX;
		++currentXp1;
	}

	return 1;
}

}
}
}

#endif

