#include <cctag/Global.hpp>
#include <cctag/toolbox.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/algebra/lapack.hpp>
#include <cctag/algebra/matrix/operation.hpp>

#include <opencv/cv.hpp>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <cmath>
#include <fstream>
#include <vector>

namespace cctag {
    namespace numerical {

        double innerProdMin(const std::vector<cctag::EdgePoint*>& filteredChildrens, double thrCosDiffMax, Point2dN<int> & p1, Point2dN<int> & p2) {
            using namespace boost::numeric;
            //using namespace cctag::numerical;

            EdgePoint* pAngle1 = NULL;
            EdgePoint* pAngle2 = NULL;

            double min = 1.1;

            double normGrad = -1;

            double distMax = 0.0;

            EdgePoint* p0 = filteredChildrens.front();

            if (filteredChildrens.size()) {

                //double normGrad = ublas::norm_2(gradient);
                //sumDeriv(0) += gradient(0)/normGrad;
                //sumDeriv(1) += gradient(1)/normGrad;

                normGrad = std::sqrt(p0->_grad.x() * p0->_grad.x() + p0->_grad.y() * p0->_grad.y());

                //CCTAG_COUT_VAR(normGrad);

                // Step 1
                double gx0 = p0->_grad.x() / normGrad;
                double gy0 = p0->_grad.y() / normGrad;

                std::vector<cctag::EdgePoint*>::const_iterator it = ++filteredChildrens.begin();

                for (; it != filteredChildrens.end(); ++it) {
                    EdgePoint* pCurrent = *it;

                    // TODO Revoir les structure de donnée pour les points 2D et définir un produit scalaire utilisé ici
                    normGrad = std::sqrt(pCurrent->_grad.x() * pCurrent->_grad.x() + pCurrent->_grad.y() * pCurrent->_grad.y());

                    double gx = pCurrent->_grad.x() / normGrad;
                    double gy = pCurrent->_grad.y() / normGrad;

                    double innerProd = gx0 * gx + gy0 * gy;

                    if (innerProd <= thrCosDiffMax)
                        return innerProd;

                    //std::cout << "innerProd : " << innerProd << std::endl;

                    if (innerProd < min) {
                        min = innerProd;
                        pAngle1 = pCurrent;
                    }

                    double dist = cctag::numerical::distancePoints2D(*p0, *pCurrent);
                    if (dist > distMax) {
                        distMax = dist;
                        p1 = *pCurrent;
                    }
                }

                normGrad = std::sqrt(pAngle1->_grad.x() * pAngle1->_grad.x() + pAngle1->_grad.y() * pAngle1->_grad.y());
                double gxmin = pAngle1->_grad.x() / normGrad;
                double gymin = pAngle1->_grad.y() / normGrad;

                // Step 2, compute the minimum inner product
                min = 1.0;
                distMax = 0.0;

                it = filteredChildrens.begin();

                //CCTAG_COUT(" 2- 2eme element" << **it);

                for (; it != filteredChildrens.end(); ++it) {
                    EdgePoint* pCurrent = *it;
                    // TODO Revoir les structure de donnée pour les point 2D et définir un produit scalaire utilisé ici
                    normGrad = std::sqrt(pCurrent->_grad.x() * pCurrent->_grad.x() + pCurrent->_grad.y() * pCurrent->_grad.y());

                    double chgx = pCurrent->_grad.x() / normGrad;
                    double chgy = pCurrent->_grad.y() / normGrad;

                    double innerProd = gxmin * chgx + gymin * chgy;

                    if (innerProd <= thrCosDiffMax)
                        return innerProd;

                    if (innerProd < min) {
                        min = innerProd;
                        pAngle2 = pCurrent;
                    }

                    double dist = cctag::numerical::distancePoints2D(p1, (Point2dN<int>)(*pCurrent));
                    if (dist > distMax) {
                        distMax = dist;
                        p2 = *pCurrent;
                    }
                }
            }

            return min;
        }

        void ellipseFitting(cctag::numerical::geometry::Ellipse& e, const std::vector< Point2dN<double> >& points) {
            std::vector<cv::Point2f> cvPoints;
            cvPoints.reserve(points.size());

            BOOST_FOREACH(const Point2dN<double> & p, points) {
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

            double angle = rR.angle * boost::math::constants::pi<double>() / 180.0;

            if ((a == 0) || (b == 0))
                CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate ellipse after cv::fitEllipse => line or point."));

            e.setParameters(Point2dN<double>(xC, yC), a, b, angle);
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

	double angle = rR.angle * boost::math::constants::pi<double>() / 180.0;

	if ( ( a == 0) || ( b == 0 ) )
		CCTAG_THROW( exception::BadHandle() << exception::dev( "Degenerate ellipse after cv::fitEllipse => line or point." ) );

	e.setParameters( Point2dN<double>( xC, yC ), a, b, angle );
}

void circleFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points) {
            using namespace boost::numeric;
            
            std::size_t nPoints = points.size();

            ublas::matrix<double, ublas::column_major> A(nPoints, 4);

            // utiliser la même matrice à chaque fois et rajouter les données.
            // Initialiser la matrice a l'exterieur et remplir ici puis inverser, idem
            // pour le fitellipse, todo@Lilian

            for (int i = 0; i < nPoints; ++i) {
                A(i, 0) = points[i]->x();
                A(i, 1) = points[i]->y();
                A(i, 2) = 1;
                A(i, 3) = points[i]->x() * points[i]->x() + points[i]->y() * points[i]->y();
            }


            ublas::matrix<double> U;
            ublas::matrix<double> V;
            ublas::diagonal_matrix<double> S;

            cctag::numerical::svd(A, U, V, S);


            //CCTAG_COUT_VAR(A);
            //CCTAG_COUT_VAR(U);
            //CCTAG_COUT_VAR(V);
            //CCTAG_COUT_VAR(S);
            //CCTAG_COUT("V(:,end) = " << V(0, 3) << " " << V(1, 3) << " " << V(2, 3) << " " << V(3, 3) << " ");

            double xC = -0.5 * V(0, 3) / V(3, 3);
            double yC = -0.5 * V(1, 3) / V(3, 3);
            double radius = sqrt(xC*xC + yC*yC - V(2, 3) / V(3, 3));

            if (radius <= 0) {
                CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate circle in circleFitting."));
            }

            e.setParameters(Point2dN<double>(xC, yC), radius, radius, 0);
        }

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::list<cctag::EdgePoint*>& points )
{
            std::vector<cv::Point2f> cvPoints;
            cvPoints.reserve(points.size());

            BOOST_FOREACH(cctag::EdgePoint * p, points) {
                cvPoints.push_back(cv::Point2f(p->x(), p->y()));
            }

            if( cvPoints.size() < 5 ) {
                std::cerr << __FILE__ << ":" << __LINE__ << " not enough points for fitEllipse" << std::endl;
            }
            cv::RotatedRect rR = cv::fitEllipse(cv::Mat(cvPoints));
            float xC = rR.center.x;
            float yC = rR.center.y;

            float b = rR.size.height / 2.f;
            float a = rR.size.width / 2.f;

            double angle = rR.angle * boost::math::constants::pi<double>() / 180.0;

            if ((a == 0) || (b == 0))
                CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate ellipse after cv::fitEllipse => line or point."));

            e.setParameters(Point2dN<double>(xC, yC), a, b, angle);
        }

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
            namespace ublas = boost::numeric::ublas;
            using namespace std;

            double a = ellipse.a();
            double b = ellipse.b();
            double angle = ellipse.angle();

            double A = -b * sin(angle) - b * cos(angle);
            double B = -a * cos(angle) + a * sin(angle);

            double t11 = atan2(-A, B);
            double t12 = t11 + M_PI;

            //A = -b*sin(teta)+b*cos(teta);
            //B = -a*cos(teta)-a*sin(teta);

            //double t21 = atan2(-A,B);
            //double t22 = t21+M_PI;

            ublas::bounded_vector<double, 3> pt1(3);
            ublas::bounded_vector<double, 3> pt2(3);

            ellipsePoint(ellipse, t11, pt1);
            ellipsePoint(ellipse, t12, pt2);

            double semiXPerm = (fabs(boost::math::round(pt1(0)) - boost::math::round(pt2(0))) - 1) * 2;
            double semiYPerm = (fabs(boost::math::round(pt1(1)) - boost::math::round(pt2(1))) - 1) * 2;

            return semiXPerm + semiYPerm;
        }

    }
}
