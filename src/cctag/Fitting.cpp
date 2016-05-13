/*------------------------------------------------------------------------------

  Copyright (c) 2012-2014 viorica patraucean (vpatrauc@gmail.com)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.


  ellipse_fit_with_gradients.c - functions to normalize a set of points and to
                                 algebraically fit an ellipse to these points,
                                 using positional and tangential constraints.

------------------------------------------------------------------------------*/





#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/utils/Defines.hpp>
#include <Eigen/SVD>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/Fitting.hpp>
#include <cmath>
#include <float.h>
#include <fstream>
#include <math.h>
#include <opencv/cv.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cctag {
namespace numerical {

#if 0
/*----------------------------------------------------------------------------*/
/** Error or exception handling: print error message and exit.
 */
void error( const char *msg )
{
  fprintf( stderr,"%s\n", msg );
  exit(EXIT_FAILURE);
}


/*----------------------------------------------------------------------------*/
/** Convert ellipse from matrix form to common form:
    ellipse = (centrex,centrey,ax,ay,orientation).
 */
static void ellipse2param( float *p, float *param)
{
  /* p = [ a,     1/2*b, 1/2*d,
           1/2*b,     c, 1/2*e,
           1/2*d, 1/2*e, f     ]; */
  float a, b, c, d, e, f;
  float thetarad, cost, sint, cos_squared, sin_squared, cos_sin;
  float Ao, Au, Av, Auu, Avv;
  float tuCentre, tvCentre, wCentre, uCentre, vCentre, Ru, Rv;

  /* Check parameters */
  if( p == NULL ) error("ellipse2param: null input ellipse matrix.");
  if( param == NULL ) error("ellipse2param: output 'param' should be non-null");

  a =   p[0];
  b = 2*p[1];
  c =   p[4];
  d = 2*p[2];
  e = 2*p[5];
  f =   p[8];

  thetarad = 0.5f*atan2(b,a-c);
  cost = cos(thetarad);
  sint = sin(thetarad);
  sin_squared = sint * sint;
  cos_squared = cost * cost;
  cos_sin = sint * cost;
  Ao  =  f;
  Au  =  d*cost + e*sint;
  Av  = -d*sint + e*cost;
  Auu = a*cos_squared + c*sin_squared + b*cos_sin;
  Avv = a*sin_squared + c*cos_squared - b*cos_sin;

  if( (Auu == 0) || (Avv==0) )
    {
      param[0] = 0;
      param[1] = 0;
      param[2] = 0;
      param[3] = 0;
      param[4] = 0;
    }
  else
    {
      tuCentre = -Au / (2.*Auu);
      tvCentre = -Av / (2.*Avv);

      wCentre  =  Ao-Auu*tuCentre*tuCentre - Avv*tvCentre * tvCentre;
      uCentre  =  tuCentre*cost - tvCentre*sint;
      vCentre  =  tuCentre*sint + tvCentre*cost;

      Ru = -wCentre / Auu;
      Rv = -wCentre / Avv;

      if( Ru>0 ) Ru =  pow( Ru,0.5f);
      else       Ru = -pow(-Ru,0.5f);

      if( Rv>0 ) Rv =  pow( Rv,0.5f);
      else       Rv = -pow(-Rv,0.5f);

      param[0] = uCentre;
      param[1] = vCentre;
      param[2] = Ru;
      param[3] = Rv;
      param[4] = thetarad;
    }
}


/*----------------------------------------------------------------------------*/
/** Compute normalisation matrix (translates and normalises a set of
    2D homogeneous points so that their centroid is at the origin and
    their mean distance from the origin is sqrt(2).
    Input points in 'reg' are in cartesian coordinates (x,y).
    Output matrix 'T' is 3 x 3.
 */
static void vgg_conditioner_from_points( float *T, float *pts, int pts_size )
{
  float mx = 0.f, my = 0.f;
  float qmean = 0.f, valx = 0.f, valy = 0.f, val;
  int i;
  float SQRT2 = sqrt(2.0);

  /* Check parameters */
  if( pts == NULL ) error("vgg_conditioner_from_points: invalid points list.");
  if( pts_size <= 0 ) error("vgg_conditioner_from_points: invalid list size.");

  /* Compute mean point */
  for( i=0; i<pts_size; i++ )
    {
      mx += pts[2*i];
      my += pts[2*i+1];
    }
  mx /= (float)pts_size; my /= (float)pts_size;

  /* Compute mean variance */
  for( i=0; i<pts_size; i++ )
    {
      valx += (pts[2*i] - mx)   * (pts[2*i] - mx);
      valy += (pts[2*i+1] - my) * (pts[2*i+1] - my);
    }
  valx = sqrt(valx);
  valy = sqrt(valy);
  qmean = (valx + valy) / 2.0;

  /* Build normalization matrix */
  val = SQRT2/qmean;
  T[1] = T[3] = 0;
  T[0] = T[4] = val;
  T[2] = -val * mx;
  T[5] = -val * my;
  T[6] = T[7] = 0;
  T[8] = 1;
}


/*----------------------------------------------------------------------------*/
/** antisym(u)  A = [ 0,-u(2),u(1); u(2),0,-u(0); -u(1),u(0),0 ];
 */
static void antisym( float *u, float *A )
{
  A[0] = A[4] = A[8] = 0;
  A[1] = -u[2];
  A[2] =  u[1];
  A[3] =  u[2];
  A[5] = -u[0];
  A[6] = -u[1];
  A[7] =  u[0];
}


/*----------------------------------------------------------------------------*/
/** Compute equations coefficients for ellipse fitting.
 */
static void get_equations( float *pts, float *grad, int pts_size, float *vgg,
                           float *buff )
{
  int i,j;
  float K[27];
  float asym[9];
  int idx;
  float crosspr[3];
  float pnormx, pnormy, dirnormx, dirnormy;

  /* Check parameters */
  if( pts == NULL ) error("get_equations: invalid points list.");
  if( grad == NULL ) error("get_equations: invalid input grad");
  if( pts_size<=0 )   error("get_equations: invalid points size.");
  if( vgg == NULL ) error("get_equations: invalid normalization matrix.");
  if( buff == NULL ) error("get_equations: invalid memory buffer.");

  /* Compute normalisation matrix */
  vgg_conditioner_from_points( vgg, pts, pts_size );

  /* Compute linear system of equations */
  for( i=0; i<pts_size; i++)
    {
      idx = i*4*6;
      /* Normalise point (pnormx,pnormy) = VGG*(x,y) */
      pnormx = vgg[0]*pts[2*i] + vgg[1]*pts[2*i+1] + vgg[2];
      pnormy = vgg[3]*pts[2*i] + vgg[4]*pts[2*i+1] + vgg[5];

      /* Normalise gradient direction (dirnormx,dirnormy) = VGG*(dx,dy) */
      dirnormx = -vgg[0] * grad[2*i+1] + vgg[1] * grad[2*i];
      dirnormy = -vgg[3] * grad[2*i+1] + vgg[4] * grad[2*i];

      /* Cross product (pnormx,pnormy) x (dirnormx,dirnormy) = tangent line */
      crosspr[0] = -dirnormy;
      crosspr[1] =  dirnormx;
      crosspr[2] =  pnormx * dirnormy - pnormy * dirnormx;

      /* Tangent equation: eq = -transpose(kron(TPts(1:3,i),antisym(l)))*J; */
      antisym(crosspr,asym);

      for (j=0;j<9;j++) K[j]    = asym[j]*pnormx;
      for (j=0;j<9;j++) K[j+9]  = asym[j]*pnormy;
      for (j=0;j<9;j++) K[j+18] = asym[j];

      buff[idx]   = - K[0];
      buff[idx+1] = -(K[3]+K[9]);
      buff[idx+2] = -(K[6]+K[18]);
      buff[idx+3] = - K[12];
      buff[idx+4] = -(K[15]+K[21]);
      buff[idx+5] = - K[24];

      buff[idx+6]   = - K[1];
      buff[idx+6+1] = -(K[4]+K[10]);
      buff[idx+6+2] = -(K[7]+K[19]);
      buff[idx+6+3] = - K[13];
      buff[idx+6+4] = -(K[16]+K[22]);
      buff[idx+6+5] = - K[25];

      buff[idx+12]   = - K[2];
      buff[idx+12+1] = -(K[5]+K[11]);
      buff[idx+12+2] = -(K[8]+K[20]);
      buff[idx+12+3] = - K[14];
      buff[idx+12+4] = -(K[17]+K[23]);
      buff[idx+12+5] = - K[26];

      /* Position equation: eq = transpose(kron(TPts(1:3,i),TPts(1:3,i)))*J; */
      buff[idx+18] = pnormx * pnormx;
      buff[idx+19] = 2 * pnormx * pnormy;
      buff[idx+20] = 2 * pnormx;
      buff[idx+21] = pnormy * pnormy;
      buff[idx+22] = 2 * pnormy;
      buff[idx+23] = 1;
    }
}


/*----------------------------------------------------------------------------*/
/** Solve linear system of equations.
 */
static void fit_ellipse( float *eq, float *vgg, int pts_size, float *param )
{
  int i,j,k;
  float A[36];

  /* Check parameters */
  if( pts_size <= 0 ) error("fit_ellipse: invalid size.");
  if( eq == NULL  ) error("fit_ellipse: invalid buffer.");
  if( vgg == NULL   ) error("fit_ellipse: invalid normalization matrix.");
  if( param == NULL ) error("fit_ellipse: param must be non null.");

  /* A = EQ'*EQ; */
  for( i=0; i<36; i++ ) A[i] = 0.f;

  for( i=0; i<6; i++ )
    for( j=0; j<6; j++ )
      for( k=0; k<4*pts_size; k++ )
        A[i*6+j] += eq[k*6+i] * eq[k*6+j];

  /* Lapack call to solve linear system */
#define SIZE6 6
  char JOBZ = 'V';
  char UPLO = 'U';
  int M = SIZE6;
  int LDA = M;
  int LWORK = 4*SIZE6;
  int INFO;
  float W[SIZE6];
  float WORK[LWORK];
  //dsyev_( &JOBZ, &UPLO, &M, A, &LDA, W, WORK, &LWORK, &INFO );


  boost::numeric::bindings::lapack::detail::syev(JOBZ, UPLO, M, A , LDA, W, WORK, LWORK, INFO);
      //{
      //  LAPACK_DSYEV (&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
      //}

  float s[9];
  s[0] = A[0];
  s[1] = s[3] = A[1];
  s[2] = s[6] = A[2];
  s[4] = A[3];
  s[5] = s[7] = A[4];
  s[8] = A[5];

  /* Apply inverse(normalisation matrix) */
  /* C = T'*[ x(1),x(2),x(3); x(2),x(4),x(5) ; x(3),x(5),x(6)]*T; */

  float C[9];
  C[0] = vgg[0]*vgg[0]*s[0] + vgg[0]*vgg[3]*s[3] +
         vgg[0]*vgg[3]*s[1] + vgg[3]*vgg[3]*s[4];
  C[1] = vgg[0]*vgg[1]*s[0] + vgg[1]*vgg[3]*s[3] +
         vgg[0]*vgg[4]*s[1] + vgg[3]*vgg[4]*s[4];
  C[2] = vgg[0]*vgg[2]*s[0] + vgg[2]*vgg[3]*s[3] +
         vgg[0]*vgg[5]*s[1] + vgg[3]*vgg[5]*s[4] + vgg[0]*s[2] + vgg[3]*s[5];

  C[3] = vgg[0]*vgg[1]*s[0] + vgg[0]*vgg[4]*s[3] +
         vgg[1]*vgg[3]*s[1] + vgg[3]*vgg[4]*s[4];
  C[4] = vgg[1]*vgg[1]*s[0] + vgg[1]*vgg[4]*s[3] +
         vgg[1]*vgg[4]*s[1] + vgg[4]*vgg[4]*s[4];
  C[5] = vgg[1]*vgg[2]*s[0] + vgg[2]*vgg[4]*s[3] +
         vgg[1]*vgg[5]*s[1] + vgg[4]*vgg[5]*s[4] + vgg[1]*s[2] + vgg[4]*s[5];

  C[6] = vgg[0]*vgg[2]*s[0] + vgg[0]*vgg[5]*s[3] +
         vgg[0]*s[6] + vgg[2]*vgg[3]*s[1] + vgg[3]*vgg[5]*s[4] + vgg[3]*s[7];
  C[7] = vgg[1]*vgg[2]*s[0] + vgg[1]*vgg[5]*s[3] +
         vgg[1]*s[6] + vgg[2]*vgg[4]*s[1] + vgg[4]*vgg[5]*s[4] + vgg[4]*s[7];
  C[8] = vgg[2]*vgg[2]*s[0] + vgg[2]*vgg[5]*s[3] + vgg[2]*s[6] +
         vgg[2]*vgg[5]*s[1] + vgg[5]*vgg[5]*s[4] + vgg[5]*s[7] + vgg[2]*s[2] +
         vgg[5]*s[5] + s[8];

  ellipse2param(C,param);
}

/*----------------------------------------------------------------------------*/
/** Algebraic ellipse fitting using positional and tangential constraints.
    pts = [x_0, y_0, x_1, y_1, ...];
    grad = [gradx_0, grady_0, gradx_1, grady_1...];
    Uses e memory buffer to compute equation coefficients. When performing
    repeatedly this operation, it is wise to reuse the same buffer, to avoid
    useless allocations/deallocations.
 */
void ellipse_fit_with_gradients( float *pts, float *grad, int pts_size,
                                 float **buff, int *size_buff_max,
                                 float *param )
{
  float vgg[9];
  /* Check parameters */
  if( pts == NULL ) error("get_equations: invalid points list.");
  if( grad == NULL ) error("get_equations: invalid input gradx");
  if( pts_size<=0 ) error("get_equations: invalid points size.");
  if( buff == NULL ) error("get_equations: invalid buffer.");
  if( *buff == NULL ) error("get_equations: invalid buffer.");

  /* If buffer too small, allocate twice the memory required at this step,
     to avoid repeating too often the memory allocation, which is expensive */
  if( pts_size * 24 > *size_buff_max )
    {
      *buff = (float*) realloc( *buff, sizeof(float) * 2 * pts_size * 24 );
      if( *buff == NULL ) error("get_equations: not enough memory.");
      *size_buff_max = 2 * pts_size * 24;
    }
  /* Compute coeffs of linear system */
  get_equations( pts, grad, pts_size, vgg, *buff );

  /* Solve linear system */
  fit_ellipse( *buff, vgg, pts_size, param );
}


void ellipseFittingWithGradientsToto( const std::vector<EdgePoint *> & vPoint, cctag::numerical::geometry::Ellipse & ellipse ){

	std::vector<float> pts;
	pts.reserve(vPoint.size()*2);
	std::vector<float> grad;
	pts.reserve(grad.size()*2);

	BOOST_FOREACH(const EdgePoint* point, vPoint){
		pts.push_back(point->x());
		pts.push_back(point->y());
		grad.push_back(point->dX());
		grad.push_back(point->dY());
	}

	std::vector<float> param;
	param.reserve(5);

	float *buff = NULL;

	buff = (float *) malloc(  sizeof(float) * 2 * pts.size() * 24 );

	int size_buff_max = 2 * pts.size() * 24;

	ellipse_fit_with_gradients( &pts[0], &grad[0], vPoint.size(),
                                 &buff, &size_buff_max,
                                 &param[0] );

	ellipse = cctag::numerical::geometry::Ellipse(Point2d<Eigen::Vector3f>(param[0],param[1]), param[2], param[3], param[4]);
}
#endif

float innerProdMin(const std::vector<cctag::EdgePoint*>& filteredChildrens, float thrCosDiffMax, Point2d<Vector3s> & p1, Point2d<Vector3s> & p2) {
            using namespace boost::numeric;
            //using namespace cctag::numerical;

            EdgePoint* pAngle1 = NULL;
            EdgePoint* pAngle2 = NULL;

            float min = 1.1;

            float normGrad = -1;

            float distMax = 0.f;

            EdgePoint* p0 = filteredChildrens.front();

            if (filteredChildrens.size()) {

                //float normGrad = ublas::norm_2(gradient);
                //sumDeriv(0) += gradient(0)/normGrad;
                //sumDeriv(1) += gradient(1)/normGrad;

                normGrad = std::sqrt(p0->dX() * p0->dX() + p0->dY() * p0->dY());

                //CCTAG_COUT_VAR(normGrad);

                // Step 1
                float gx0 = p0->dX() / normGrad;
                float gy0 = p0->dY() / normGrad;

                std::vector<cctag::EdgePoint*>::const_iterator it = ++filteredChildrens.begin();

                for (; it != filteredChildrens.end(); ++it) {
                    EdgePoint* pCurrent = *it;

                    // TODO Revoir les structure de donnée pour les points 2D et définir un produit scalaire utilisé ici
                    normGrad = std::sqrt(pCurrent->dX() * pCurrent->dX() + pCurrent->dY() * pCurrent->dY());

                    float gx = pCurrent->dX() / normGrad;
                    float gy = pCurrent->dY() / normGrad;

                    float innerProd = gx0 * gx + gy0 * gy;

                    if (innerProd <= thrCosDiffMax)
                        return innerProd;

                    //std::cout << "innerProd : " << innerProd << std::endl;

                    if (innerProd < min) {
                        min = innerProd;
                        pAngle1 = pCurrent;
                    }

                    float dist = cctag::numerical::distancePoints2D(*p0, *pCurrent);
                    if (dist > distMax) {
                        distMax = dist;
                        p1 = *pCurrent;
                    }
                }

                normGrad = std::sqrt(pAngle1->dX() * pAngle1->dX() + pAngle1->dY() * pAngle1->dY());
                float gxmin = pAngle1->dX() / normGrad;
                float gymin = pAngle1->dY() / normGrad;

                // Step 2, compute the minimum inner product
                min = 1.f;
                distMax = 0.f;

                it = filteredChildrens.begin();

                //CCTAG_COUT(" 2- 2eme element" << **it);

                for (; it != filteredChildrens.end(); ++it) {
                    EdgePoint* pCurrent = *it;
                    // TODO Revoir les structure de donnée pour les point 2D et définir un produit scalaire utilisé ici
                    normGrad = std::sqrt(pCurrent->dX() * pCurrent->dX() + pCurrent->dY() * pCurrent->dY());

                    float chgx = pCurrent->dX() / normGrad;
                    float chgy = pCurrent->dY() / normGrad;

                    float innerProd = gxmin * chgx + gymin * chgy;

                    if (innerProd <= thrCosDiffMax)
                        return innerProd;

                    if (innerProd < min) {
                        min = innerProd;
                        pAngle2 = pCurrent;
                    }

                    float dist = cctag::numerical::distancePoints2D(p1, (Point2d<Vector3s>)(*pCurrent));
                    if (dist > distMax) {
                        distMax = dist;
                        p2 = *pCurrent;
                    }
                }
            }

            return min;
        }

void ellipseFitting(cctag::numerical::geometry::Ellipse& e, const std::vector< Point2d<Eigen::Vector3f> >& points) {
            std::vector<cv::Point2f> cvPoints;
            cvPoints.reserve(points.size());

            BOOST_FOREACH(const Point2d<Eigen::Vector3f> & p, points) {
                cvPoints.push_back(cv::Point2f(p.x(), p.y()));
            }

            if( cvPoints.size() < 5 ) {
                std::cerr << __FILE__ << ":" << __LINE__ << " not enough points for fitEllipse" << std::endl;
            }
            cv::RotatedRect rR = cv::fitEllipse(cv::Mat(cvPoints));
            float xC = rR.center.x;
            float yC = rR.center.y;

            float b = rR.size.height / 2.f;
            float a = rR.size.width / 2.f;

            float angle = rR.angle * boost::math::constants::pi<float>() / 180.0;

            if ((a == 0) || (b == 0))
                CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate ellipse after cv::fitEllipse => line or point."));

            e.setParameters(Point2d<Eigen::Vector3f>(xC, yC), a, b, angle);
        }
        
void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points )
{
	std::vector<cv::Point2f> cvPoints;
	cvPoints.reserve( points.size() );
	BOOST_FOREACH( cctag::EdgePoint * p, points )
	{
		cvPoints.push_back( cv::Point2f( p->x(), p->y() ) );
            }

    if( cvPoints.size() < 5 ) {
        std::cerr << __FILE__ << ":" << __LINE__ << " not enough points for fitEllipse" << std::endl;
    }
	cv::RotatedRect rR = cv::fitEllipse( cv::Mat( cvPoints ) );
	float xC           = rR.center.x;
	float yC           = rR.center.y;

	float b = rR.size.height / 2.f;
	float a = rR.size.width / 2.f;

	float angle = rR.angle * boost::math::constants::pi<float>() / 180.0;

	if ( ( a == 0) || ( b == 0 ) )
		CCTAG_THROW( exception::BadHandle() << exception::dev( "Degenerate ellipse after cv::fitEllipse => line or point." ) );

	e.setParameters( Point2d<Eigen::Vector3f>( xC, yC ), a, b, angle );
}

void circleFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points) {
            using namespace boost::numeric;
            
            std::size_t nPoints = points.size();

            Eigen::MatrixXf A(nPoints, 4);

            // utiliser la même matrice à chaque fois et rajouter les données.
            // Initialiser la matrice a l'exterieur et remplir ici puis inverser, idem
            // pour le fitellipse, todo@Lilian

            for (int i = 0; i < nPoints; ++i) {
                A(i, 0) = points[i]->x();
                A(i, 1) = points[i]->y();
                A(i, 2) = 1;
                A(i, 3) = points[i]->x() * points[i]->x() + points[i]->y() * points[i]->y();
            }
            
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto V = svd.matrixV();

            //CCTAG_COUT_VAR(A);
            //CCTAG_COUT_VAR(U);
            //CCTAG_COUT_VAR(V);
            //CCTAG_COUT_VAR(S);
            //CCTAG_COUT("V(:,end) = " << V(0, 3) << " " << V(1, 3) << " " << V(2, 3) << " " << V(3, 3) << " ");

            float xC = -0.5f * V(0, 3) / V(3, 3);
            float yC = -0.5f * V(1, 3) / V(3, 3);
            float radius = sqrt(xC*xC + yC*yC - V(2, 3) / V(3, 3));

            if (radius <= 0) {
                CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate circle in circleFitting."));
            }

            e.setParameters(Point2d<Eigen::Vector3f>(xC, yC), radius, radius, 0);
        }

#if 0
        bool matrixFromFile(const std::string& filename, std::list<cctag::EdgePoint>& edgepoints) {
            std::ifstream ifs(filename.c_str());

            if (!ifs) {
                throw ( "Cannot open file");
            }

            std::stringstream oss;
            oss << ifs.rdbuf();

            if (!ifs && !ifs.eof()) {
                throw ( "Error reading file");
            }
            std::string str = oss.str();

            std::vector<std::string> lines;
            boost::split(lines, str, boost::is_any_of("\n"));
            for (std::vector<std::string>::iterator it = lines.begin(); it != lines.end(); ++it) {
                std::vector<std::string> xy;
                boost::split(xy, *it, boost::is_any_of(", "));
                if (xy.size() == 2) {
                    edgepoints.push_back(cctag::EdgePoint(boost::lexical_cast<int>(xy[0]), boost::lexical_cast<int>(xy[1]), 0, 0));
                }
            }

            return true;
        }

        int discreteEllipsePerimeter(const cctag::numerical::geometry::Ellipse& ellipse) {
            using namespace std;

            float a = ellipse.a();
            float b = ellipse.b();
            float angle = ellipse.angle();

            float A = -b * sin(angle) - b * cos(angle);
            float B = -a * cos(angle) + a * sin(angle);

            float t11 = atan2(-A, B);
            float t12 = t11 + M_PI;

            //A = -b*sin(teta)+b*cos(teta);
            //B = -a*cos(teta)-a*sin(teta);

            //float t21 = atan2(-A,B);
            //float t22 = t21+M_PI;

            Eigen::Vector3f pt1, pt2;
            ellipsePoint(ellipse, t11, pt1);
            ellipsePoint(ellipse, t12, pt2);

            float semiXPerm = (fabs(boost::math::round(pt1(0)) - boost::math::round(pt2(0))) - 1) * 2;
            float semiYPerm = (fabs(boost::math::round(pt1(1)) - boost::math::round(pt2(1))) - 1) * 2;

            return semiXPerm + semiYPerm;
        }
#endif
} // namespace numerical
} // namespace cctag
