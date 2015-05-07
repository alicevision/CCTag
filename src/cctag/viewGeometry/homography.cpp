#include "homography.hpp"

//pour le calibrage
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

#include <iostream>

namespace cctag {
namespace viewGeometry {

//transforme la matrice d'une ellipse en ses paramètres correspondants
void ellipse_to_params( CvMat* E, double param[5] )
{
	//   [     a, 1/2*b, 1/2*d;
	//    1/2*b,     c, 1/2*e;
	//    1/2*d, 1/2*e,     f ];
	double a = (double)( cvmGet( E, 0, 0 ) ), b = 2 * (double)( cvmGet( E, 0, 1 ) ), c = (double)( cvmGet( E, 1, 1 ) ), d = 2 * (double)( cvmGet( E, 0, 2 ) ), e = 2 * (double)( cvmGet( E, 1, 2 ) ), f = (double)( cvmGet( E, 2, 2 ) ), thetarad, cost, sint, cos_squared, sin_squared, cos_sin, Ao, Au, Av, Auu, Avv, tuCentre, tvCentre, wCentre, uCentre, vCentre, Ru, Rv;

	if( a - c == 0 )
		thetarad = M_PI / 2.;
	else
		thetarad = 0.5 * atan( b / ( a - c ) );
	cost        = cos( thetarad );
	sint        = sin( thetarad );
	sin_squared = sint * sint;
	cos_squared = cost * cost;
	cos_sin     = sint * cost;
	Ao          = f;
	Au          = d * cost + e * sint;
	Av          = -d * sint + e * cost;
	Auu         = a * cos_squared + c * sin_squared + b * cos_sin;
	Avv         = a * sin_squared + c * cos_squared - b * cos_sin;

	if( Auu == 0 || Avv == 0 )
	{
		param[0] = 0; param[1] = 0; param[2] = 0; param[3] = 0; param[4] = 0;
	}
	else
	{
		tuCentre = -Au / ( 2. * Auu );
		tvCentre = -Av / ( 2. * Avv );
		wCentre  = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;
		uCentre  = tuCentre * cost - tvCentre * sint;
		vCentre  = tuCentre * sint + tvCentre * cost;
		Ru       = -wCentre / Auu;
		Rv       = -wCentre / Avv;
		if( Ru > 0 )
			Ru = pow( Ru, 0.5 );
		else
			Ru = -pow( -Ru, 0.5 );
		if( Rv > 0 )
			Rv = pow( Rv, 0.5 );
		else
			Rv = -pow( -Rv, 0.5 );
		param[0] = uCentre; param[1] = vCentre; param[2] = Ru; param[3] = Rv; param[4] = thetarad;
	}
}

//transforme les paramètres d'une ellipse en sa matrice correspondante
void param_to_ellipse( const double x, const double y, const double largeur, const double hauteur, const double angle, CvMat* Ellipse )
{
	CvMat* D           = cvCreateMat( 3, 3, CV_64FC1 );
	CvMat* M           = cvCreateMat( 3, 3, CV_64FC1 );
	CvMat* invM        = cvCreateMat( 3, 3, CV_64FC1 );
	CvMat* transp_invM = cvCreateMat( 3, 3, CV_64FC1 );
	CvMat* M2          = cvCreateMat( 3, 3, CV_64FC1 );
	double diag[]      = { 1 / pow( largeur, 2 ), 0, 0, 0, 1 / pow( hauteur, 2 ), 0, 0, 0, -1 };

	cvInitMatHeader( D, 3, 3, CV_64FC1, diag );
	double param[] = { cos( angle ), -sin( angle ), x, sin( angle ), cos( angle ), y, 0, 0, 1 };
	cvInitMatHeader( M, 3, 3, CV_64FC1, param );
	cvInvert( M, invM );
	cvTranspose( invM, transp_invM );
	cvMatMul( transp_invM, D, M2 );
	cvMatMul( M2, invM, Ellipse );

	//libère la mémoire
	cvReleaseMat( &D ); cvReleaseMat( &M ); cvReleaseMat( &invM ); cvReleaseMat( &transp_invM ); cvReleaseMat( &M2 );
}

//retrouve l'homographie faisant passer le plan de la caméra au plan de la mire
//E0 et E1 correspondent aux matrices des deux ellipses, images de cercles concentriques
//E0 correspond à la matrice de l'ellipse la plus grande
CvMat retrouve_homographie( const CvMat* E0, const CvMat* E1, const int largeur, const int hauteur )
{
	//déclaration des variables
	double d00, d10, d20, d01, d11, d21, l2[30], d, s, norme_fro = 0;
	double params[5];
	CvMat* M   = cvCreateMat( 5, 6, CV_64FC1 ),       * T = cvCreateMat( 3, 3, CV_64FC1 ),       * inv_T = cvCreateMat( 3, 3, CV_64FC1 ),
	* transp_T = cvCreateMat( 3, 3, CV_64FC1 ),    * temp = cvCreateMat( 3, 3, CV_64FC1 ),    * temp2 = cvCreateMat( 3, 3, CV_64FC1 ),
	* B        = cvCreateMat( 3, 3, CV_64FC1 ),       * inv_A2 = cvCreateMat( 3, 3, CV_64FC1 ),  * D = cvCreateMat( 3, 1, CV_64FC1 ),
	* Vk       = cvCreateMat( 3, 1, CV_64FC1 ),      * linf = cvCreateMat( 3, 1, CV_64FC1 ),    * transp_linf = cvCreateMat( 1, 3, CV_64FC1 ),
	* inv_A    = cvCreateMat( 3, 3, CV_64FC1 ),   * temp3 = cvCreateMat( 1, 3, CV_64FC1 ),   * temp4 = cvCreateMat( 1, 1, CV_64FC1 ),
	* L        = cvCreateMat( 3, 3, CV_64FC1 ),       * Diag = cvCreateMat( 2, 2, CV_64FC1 ),    * L01 = cvCreateMat( 3, 2, CV_64FC1 ),
	* DI       = cvCreateMat( 3, 2, CV_64FC1 ),      * Vend = cvCreateMat( 6, 1, CV_64FC1 ),    * x_TLS = cvCreateMat( 6, 1, CV_64FC1 ),
	* CDCP     = cvCreateMat( 3, 3, CV_64FC1 ),    * BB = cvCreateMat( 3, 2, CV_64FC1 ),      * T1 = cvCreateMat( 2, 2, CV_64FC1 ),
	* A0       = cvCreateMat( 3, 3, CV_64FC1 ),      * A1 = cvCreateMat( 3, 3, CV_64FC1 ),      * H2 = cvCreateMat( 3, 3, CV_64FC1 ),
	* S, * U, * V;

	double l[] = { 4 * hauteur, 0., largeur / 2., 0, 4 * hauteur, hauteur / 2., 0., 0., 1. };

	cvInitMatHeader( T, 3, 3, CV_64FC1, l );
	cvInvert( T, inv_T );
	cvTranspose( T, transp_T );
	cvMatMul( transp_T, E0, temp );
	cvMatMul( temp, T, temp2 );
	d = cvDet( temp2 );
	s = ( d < 0 ? -1 / pow( -d, 1. / 3. ) : 1 / pow( d, 1. / 3. ) );
	cvConvertScale( temp2, A0, s );
	cvMatMul( transp_T, E1, temp );
	cvMatMul( temp, T, temp2 );
	d = cvDet( temp2 );
	s = ( d < 0 ? -1 / pow( -d, 1. / 3. ) : 1 / pow( d, 1. / 3. ) );
	cvConvertScale( temp2, A1, s );

	//calcul des valeurs propres et des vecteurs propres de B

	cvInvert( A1, inv_A2 );
	cvMatMul( A0, inv_A2, B );
	gsl_matrix* LLL = gsl_matrix_alloc( 3, 3 );
	for( int k = 0; k < 3; k++ )
		for( int l = 0; l < 3; l++ )
			gsl_matrix_set( LLL, k, l, cvmGet( B, k, l ) );
	gsl_vector_complex* eval         = gsl_vector_complex_alloc( 3 );
	gsl_matrix_complex* evec         = gsl_matrix_complex_alloc( 3, 3 );
	gsl_eigen_nonsymmv_workspace* WW = gsl_eigen_nonsymmv_alloc( 3 );
	gsl_eigen_nonsymmv( LLL, eval, evec, WW );
	gsl_eigen_nonsymmv_free( WW );
	gsl_eigen_nonsymmv_sort( eval, evec, GSL_EIGEN_SORT_ABS_DESC );
	gsl_complex eval_i             = gsl_vector_complex_get( eval, 0 );
	gsl_vector_complex_view evec_i = gsl_matrix_complex_column( evec, 0 );
	for( int l = 0; l < 3; l++ )
	{
		gsl_complex z = gsl_vector_complex_get( &evec_i.vector, l );
		CV_MAT_ELEM( *D, double, l, 0 ) = GSL_REAL( z );
	}
	gsl_vector_complex_free( eval );
	gsl_matrix_complex_free( evec );
	cvMatMul( inv_A2, D, Vk );

	cvConvertScale( Vk, Vk, 1. / cvmGet( Vk, 2, 0 ) );

	cvMatMul( A0, Vk, linf );
	cvTranspose( linf, transp_linf );
	cvInvert( A0, inv_A );
	cvMatMul( transp_linf, inv_A, temp3 );
	cvMatMul( temp3, linf, temp4 );
	cvMatMul( linf, transp_linf, temp );
	cvConvertScale( temp, temp2, 1. / cvmGet( temp4, 0, 0 ) );
	cvSub( A0, temp2, temp );
	S = cvCreateMat( 3, 3, CV_64FC1 );
	U = cvCreateMat( 3, 3, CV_64FC1 );
	V = cvCreateMat( 3, 3, CV_64FC1 );

	std::cout << __FILE__ << " : " << __LINE__ << std::endl;
	cvSVD( temp, S, U, V ); /// todo@lilian
	std::cout << __FILE__ << " : " << __LINE__ << std::endl;

	CV_MAT_ELEM( *Diag, double, 0, 0 ) = sqrt( cvmGet( S, 0, 0 ) );
	CV_MAT_ELEM( *Diag, double, 1, 0 ) = 0;
	CV_MAT_ELEM( *Diag, double, 0, 1 ) = 0;
	CV_MAT_ELEM( *Diag, double, 1, 1 ) = sqrt( cvmGet( S, 1, 1 ) );
	cvGetCols( U, L01, 0, 2 );
	cvMatMul( L01, Diag, DI );
	d00    = cvmGet( DI, 0, 0 ); d10 = cvmGet( DI, 1, 0 ); d20 = cvmGet( DI, 2, 0 );
	d01    = cvmGet( DI, 0, 1 ); d11 = cvmGet( DI, 1, 1 ); d21 = cvmGet( DI, 2, 1 );
	l2[0]  = d01 * d00;          l2[1] = d01 * d10 + d11 * d00;    l2[2] = d01 * d20 + d21 * d00;     l2[3] = d11 * d10;      l2[4] = d11 * d20 + d21 * d10;
	l2[5]  = d21 * d20;          l2[6] = pow( d00, 2 ) - pow( d01, 2 );  l2[7] = 2 * d00 * d10 - 2 * d01 * d11; l2[8] = -2 * d01 * d21 + 2 * d00 * d20; l2[9] = -pow( d11, 2 ) + pow( d10, 2 );
	l2[10] = 2 * d10 * d20 - 2 * d11 * d21; l2[11] = pow( d20, 2 ) - pow( d21, 2 ); l2[12] = cvmGet( linf, 0, 0 );   l2[13] = cvmGet( linf, 1, 0 );    l2[14] = cvmGet( linf, 2, 0 );
	l2[15] = 0;           l2[16] = 0;             l2[17] = 0;          l2[18] = 0;           l2[19] = cvmGet( linf, 0, 0 );
	l2[20] = 0;           l2[21] = cvmGet( linf, 1, 0 );      l2[22] = cvmGet( linf, 2, 0 );   l2[23] = 0;           l2[24] = 0;
	l2[25] = 0;           l2[26] = cvmGet( linf, 0, 0 );      l2[27] = 0;          l2[28] = cvmGet( linf, 1, 0 );    l2[29] = cvmGet( linf, 2, 0 );
	cvInitMatHeader( M, 5, 6, CV_64FC1, l2 );

	S = cvCreateMat( 5, 6, CV_64FC1 );
	U = cvCreateMat( 5, 5, CV_64FC1 );
	V = cvCreateMat( 6, 6, CV_64FC1 );
	cvSVD( M, S, U, V );
	cvGetCol( V, Vend, 5 );
	cvConvertScale( Vend, x_TLS, 1. );
	double l4[] = { cvmGet( x_TLS, 0, 0 ), cvmGet( x_TLS, 1, 0 ), cvmGet( x_TLS, 2, 0 ), cvmGet( x_TLS, 1, 0 ), cvmGet( x_TLS, 3, 0 ), cvmGet( x_TLS, 4, 0 ), cvmGet( x_TLS, 2, 0 ), cvmGet( x_TLS, 4, 0 ), cvmGet( x_TLS, 5, 0 ) };
	cvInitMatHeader( CDCP, 3, 3, CV_64FC1, l4 );
	for( int j = 0; j < 3; j++ )
		for( int k = 0; k < 3; k++ )
			norme_fro += pow( cvmGet( CDCP, j, k ), 2. );
	norme_fro = pow( norme_fro, 0.5 );
	cvConvertScale( CDCP, CDCP, 1. / norme_fro );
	S = cvCreateMat( 3, 3, CV_64FC1 );
	U = cvCreateMat( 3, 3, CV_64FC1 );
	V = cvCreateMat( 3, 3, CV_64FC1 );
	cvSVD( CDCP, S, U, V );
	CV_MAT_ELEM( *S, double, 0, 0 ) = pow( cvmGet( S, 0, 0 ), 0.5 );
	CV_MAT_ELEM( *S, double, 1, 1 ) = pow( cvmGet( S, 1, 1 ), 0.5 );
	CV_MAT_ELEM( *S, double, 2, 2 ) = pow( cvmGet( S, 2, 2 ), 0.5 );
	cvMatMul( U, S, V );
	cvGetCols( V, BB, 0, 2 );
	CV_MAT_ELEM( *T1, double, 0, 0 ) = -cvmGet( BB, 1, 1 );
	CV_MAT_ELEM( *T1, double, 0, 1 ) = cvmGet( BB, 1, 0 );
	CV_MAT_ELEM( *T1, double, 1, 0 ) = cvmGet( BB, 1, 0 );
	CV_MAT_ELEM( *T1, double, 1, 1 ) = cvmGet( BB, 1, 1 );
	cvMatMul( BB, T1, BB );
	cvConvertScale( BB, BB, 1. / cvmGet( BB, 2, 0 ) );
	double l5[] = { cvmGet( BB, 0, 0 ), cvmGet( BB, 0, 1 ), -cvmGet( BB, 1, 1 ), 0., cvmGet( BB, 1, 1 ), cvmGet( BB, 0, 1 ) - cvmGet( BB, 0, 0 ) * cvmGet( BB, 2, 1 ), 1., cvmGet( BB, 2, 1 ), cvmGet( BB, 0, 0 ) * cvmGet( BB, 1, 1 ) };
	cvInitMatHeader( L, 3, 3, CV_64FC1, l5 );
	cvMatMul( T, L, L );

	d = cvDet( L );
	s = ( d < 0 ? -1 / pow( -d, 1. / 3. ) : 1 / pow( d, 1. / 3. ) );
	cvConvertScale( L, H2, s );

	//transforme l'homographie de façon à ce que les centres des cercles concentriques soient en (0,0) et que le rayon du plus grand cercle soit 1
	cvConvertScale( E0, A0, 1. );
	cvConvertScale( E1, A1, 1. );
	cvInvert( H2, temp ); cvTranspose( H2, temp2 ); cvTranspose( temp, B );
	cvMatMul( temp2, A0, A0 ); cvMatMul( A0, H2, A0 );
	cvMatMul( temp2, A1, A1 ); cvMatMul( A1, H2, A1 );
	ellipse_to_params( A0, params );
	CV_MAT_ELEM( *temp, double, 0, 0 ) = ( params[2] + params[3] ) / 2.; CV_MAT_ELEM( *temp, double, 0, 1 ) = 0.; CV_MAT_ELEM( *temp, double, 0, 2 ) = params[0] / 2.;
	CV_MAT_ELEM( *temp, double, 1, 0 ) = 0.; CV_MAT_ELEM( *temp, double, 1, 1 ) = ( params[2] + params[3] ) / 2.; CV_MAT_ELEM( *temp, double, 1, 2 ) = params[1] / 2.;
	CV_MAT_ELEM( *temp, double, 2, 0 ) = 0.; CV_MAT_ELEM( *temp, double, 2, 1 ) = 0.;              CV_MAT_ELEM( *temp, double, 2, 2 ) = 1.;
	ellipse_to_params( A1, params );
	CV_MAT_ELEM( *temp, double, 0, 2 ) += params[0] / 2.; CV_MAT_ELEM( *temp, double, 1, 2 ) += params[1] / 2.;
	cvMatMul( H2, temp, H2 );
	norme_fro = pow( pow( cvmGet( H2, 0, 0 ), 2. ) + pow( cvmGet( H2, 0, 1 ), 2. ) + pow( cvmGet( H2, 0, 2 ), 2. ) + pow( cvmGet( H2, 1, 0 ), 2. ) + pow( cvmGet( H2, 1, 1 ), 2. ) + pow( cvmGet( H2, 1, 2 ), 2. ) + pow( cvmGet( H2, 2, 0 ), 2. ) + pow( cvmGet( H2, 2, 1 ), 2. ) + pow( cvmGet( H2, 2, 2 ), 2. ), 0.5 );
	cvConvertScale( H2, H2, 1. / norme_fro );

	//	H.push_back( *H2 );
	CvMat result( *H2 );

	//std::cout << "E1 :" << std::endl;
	//affiche_mat(E0);
	//std::cout << "E2 :" << std::endl;
	//affiche_mat(E1);
	//	std::cout << " Frame : " << ( H.size() - 1 ) / 2 << std::endl;

	// libère la mémoire
	cvReleaseMat( &M ); cvReleaseMat( &T ); cvReleaseMat( &transp_T ); cvReleaseMat( &inv_T ); cvReleaseMat( &temp ); cvReleaseMat( &temp2 ); cvReleaseMat( &B ); cvReleaseMat( &inv_A2 ); cvReleaseMat( &D ); cvReleaseMat( &Vk ); cvReleaseMat( &linf ); cvReleaseMat( &transp_linf ); cvReleaseMat( &inv_A ); cvReleaseMat( &temp3 ); cvReleaseMat( &temp4 ); cvReleaseMat( &Diag ); cvReleaseMat( &L ); cvReleaseMat( &L01 ); cvReleaseMat( &DI ); cvReleaseMat( &Vend ); cvReleaseMat( &x_TLS ); cvReleaseMat( &CDCP ); cvReleaseMat( &BB ); cvReleaseMat( &T1 ); cvReleaseMat( &U ); cvReleaseMat( &S ); cvReleaseMat( &V ); cvReleaseMat( &A0 ); cvReleaseMat( &A1 ); //cvReleaseMat(&H2);
	return result;
}

} // namespace viewGeometry
} // namespace cctag
