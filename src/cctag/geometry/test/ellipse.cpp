#define BOOST_TEST_MODULE testEllipse

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <cctag/geometry/Point.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <Eigen/Geometry>
#include <boost/math/constants/constants.hpp>

using Point3f = cctag::Point2d<Eigen::Vector3f>;

BOOST_AUTO_TEST_SUITE(test_ellipse)

BOOST_AUTO_TEST_CASE(test_ellipse_representation)
{
    const float aGT = 5.0f;
    const float bGT = 3.0f;
    const float cxGT = 16.0f;
    const float cyGT = -4.0f;

    // some points satisfying x^2/aGT^2 + y^2/bGT^2 = 1
    std::vector<Point3f> pts;
    pts.emplace_back( 5.0000f, .0f);
    pts.emplace_back( 4.7553f, 0.9271f);
    pts.emplace_back( 4.0451f, 1.7634f);
    pts.emplace_back( 2.9389f, 2.4271f);
    pts.emplace_back( 1.5451f, 2.8532f);
    pts.emplace_back( 0.0000f, 3.0000f);
    pts.emplace_back(-1.5451f, 2.8532f);
    pts.emplace_back(-2.9389f, 2.4271f);
    pts.emplace_back(-4.0451f, 1.7634f);
    pts.emplace_back(-4.7553f, 0.9271f);
    pts.emplace_back(-5.0000f, .0f);
    pts.emplace_back(-4.7553f, -0.9271f);
    pts.emplace_back(-4.0451f, -1.7634f);
    pts.emplace_back(-2.9389f, -2.4271f);
    pts.emplace_back(-1.5451f, -2.8532f);
    pts.emplace_back(-0.0000f, -3.0000f);
    pts.emplace_back( 1.5451f, -2.8532f);
    pts.emplace_back( 2.9389f, -2.4271f);
    pts.emplace_back( 4.0451f, -1.7634f);
    pts.emplace_back( 4.7553f, -0.9271f);
    pts.emplace_back( 5.0000f, .0f);

    const float pi_f = boost::math::constants::pi<float>();
    const float pi_2_f = boost::math::constants::half_pi<float>();
    const std::size_t numTrials = 10;
    const float step = pi_f/numTrials;

    for(std::size_t i = 0; i < numTrials; ++i)
    {
        const float angleGT = i*step;
        using namespace cctag::numerical::geometry;
        cctag::numerical::geometry::Ellipse el(Point3f(cxGT, cyGT), aGT, bGT, angleGT); // avoid conflict with wingdi.h on windows

        // checking getters
        BOOST_CHECK(std::fabs(aGT - el.a()) < 0.00001);
        BOOST_CHECK(std::fabs(bGT - el.b()) < 0.00001);
        BOOST_CHECK(std::fabs(angleGT - el.angle()) < 0.00001);
        BOOST_CHECK(std::fabs(cxGT - el.center()(0)) < 0.00001);
        BOOST_CHECK(std::fabs(cyGT - el.center()(1)) < 0.00001);

        const auto &conic = el.matrix();

        const Eigen::Rotation2D<float> rotation{angleGT};
        Eigen::Matrix3f transform;
        transform.setIdentity();
        transform.topLeftCorner<2,2>() = rotation.toRotationMatrix();
        transform(0, 2) = cxGT;
        transform(1, 2) = cyGT;

        // check the (transformed) points satisfy the equation p'*C*p = 0
        for (const auto &p : pts)
        {
            const float result = p.transpose() * transform.transpose() * conic * transform * p;
            BOOST_CHECK(std::fabs(result) < 0.001);
        }

        // Test building the ellipse form a matrix
        cctag::numerical::geometry::Ellipse fromMatrix(conic);

        // major axis aligned with y-axis choice, so a and b are swapped and the angle is 90deg away
        BOOST_CHECK(std::fabs(fromMatrix.b() - aGT) < 0.001);
        BOOST_CHECK(std::fabs(fromMatrix.a() - bGT) < 0.001);
        BOOST_CHECK(std::fabs(fromMatrix.angle() - angleGT) - pi_2_f < 0.001);

        // test decomposition
        Ellipse::Matrix canonic;
        Ellipse::Matrix primal;
        Ellipse::Matrix dual;
        el.getCanonicForm(canonic, primal, dual);

        // test primal and dual are inverse
        const Ellipse::Matrix identity = primal * dual;
        for(Ellipse::Matrix::Index r = 0; r < identity.rows(); ++r)
        {
            for(Ellipse::Matrix::Index c = 0; c < identity.cols(); ++c)
            {
                if(r == c)
                {
                    BOOST_CHECK(std::fabs(identity(r, c) - 1) < 0.001);
                }
                else
                {
                    BOOST_CHECK(std::fabs(identity(r, c)) < 0.001);
                }
            }
        }

        // test points
        for (const auto &p : pts)
        {
            // since the axis are inverted, invert also the points
            const Point3f point{p.y(), p.x()};
            const float result = point.transpose() * canonic  * point;
            BOOST_CHECK(std::fabs(result) < 0.001);
        }

        // test building from canonical form
        cctag::numerical::geometry::Ellipse fromCanonic{canonic};
        BOOST_CHECK(std::fabs(fromCanonic.b() - aGT) < 0.001);
        BOOST_CHECK(std::fabs(fromCanonic.a() - bGT) < 0.001);
        BOOST_CHECK(std::fabs(fromCanonic.angle()) < 0.0001);
        // in canonical form the center is the origin
        BOOST_CHECK(std::fabs(fromCanonic.center()(0)) < 0.0001);
        BOOST_CHECK(std::fabs(fromCanonic.center()(1)) < 0.0001);
    }
}

BOOST_AUTO_TEST_SUITE_END()

