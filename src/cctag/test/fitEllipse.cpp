#define BOOST_TEST_MODULE testFitEllipse

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <cctag/geometry/Point.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/Fitting.hpp>
#include <Eigen/Dense>

using Point3f = cctag::Point2d<Eigen::Vector3f>;


/**
 * @brief Check whether a point is inside of the ellipse
 * @param[in] pt Point to test
 * @param[out] el Ellipse
 * @return true if pt lies inside the ellipse
 */
bool check_pt_in_ellipse(const Point3f& pt, const cctag::numerical::geometry::Ellipse& el)
{
    // need to be done like this because ctor Point2d(Container) applies toNonHomogen
    // to the result that has the third coordinate w() = 0
    const Point3f to_pt = Point3f(pt.x() - el.center().x(), pt.y() - el.center().y());
    const double pt_angle = std::atan2(to_pt.y(), to_pt.x());
    const double el_angle = el.angle();
    const double x_dist = el.a() * std::cos(pt_angle + el_angle);
    const double y_dist = el.b() * std::sin(pt_angle + el_angle);
    const double el_dist = std::hypot(x_dist, y_dist);
    return to_pt.norm() < el_dist;
}

/**
 * @brief Compute the ellipse fitting the points and test whether the center of mass
 * of the points lies inside the ellipse
 * @param pts[in] The input points
 * @return true if the center of mass of the points lies inside the estimated ellipse
 */
bool fit_and_check_ellipse(const std::vector<Point3f>& pts)
{
    cctag::numerical::geometry::Ellipse ellipse;
    cctag::numerical::ellipseFitting(ellipse, pts);

    Point3f mass_center{.0f, .0f};
    for (const auto &pt : pts)
    {
        mass_center += (pt / (float)pts.size());
    }

    return check_pt_in_ellipse(mass_center, ellipse);
}

BOOST_AUTO_TEST_SUITE(test_ellipseFitting)

BOOST_AUTO_TEST_CASE(test_pixel_int)
{
    std::vector<Point3f> pts;
    pts.reserve(12);
    pts.emplace_back(327.f, 317.f);
    pts.emplace_back(328.f, 316.f);
    pts.emplace_back(329.f, 315.f);
    pts.emplace_back(330.f, 314.f);
    pts.emplace_back(331.f, 314.f);
    pts.emplace_back(332.f, 314.f);
    pts.emplace_back(333.f, 315.f);
    pts.emplace_back(333.f, 316.f);
    pts.emplace_back(333.f, 317.f);
    pts.emplace_back(333.f, 318.f);
    pts.emplace_back(333.f, 319.f);
    pts.emplace_back(333.f, 320.f);

    BOOST_CHECK(fit_and_check_ellipse(pts));
}

BOOST_AUTO_TEST_CASE(test_pixel_int2)
{
    std::vector<Point3f> pts;
    pts.reserve(12);
    pts.emplace_back(327.f, 317.f);
    pts.emplace_back(328.f, 316.f);
    pts.emplace_back(329.f, 315.f);
    pts.emplace_back(330.f, 314.f);
    pts.emplace_back(331.f, 314.f);
    pts.emplace_back(332.f, 314.f);
    pts.emplace_back(333.f, 315.f);
    pts.emplace_back(333.f, 316.f);
    pts.emplace_back(333.f, 317.f);
    pts.emplace_back(333.f, 318.f);
    pts.emplace_back(333.f, 319.f);
    pts.emplace_back(333.f, 320.f);

    BOOST_CHECK(fit_and_check_ellipse(pts));
}

BOOST_AUTO_TEST_CASE(test_float1)
{
    std::vector<Point3f> pts;
    pts.reserve(10);
    pts.emplace_back(924.784f, 764.160f);
    pts.emplace_back(928.388f, 615.903f);
    pts.emplace_back(847.400f, 888.014f);
    pts.emplace_back(929.406f, 741.675f);
    pts.emplace_back(904.564f, 825.605f);
    pts.emplace_back(926.742f, 760.746f);
    pts.emplace_back(863.479f, 873.406f);
    pts.emplace_back(910.987f, 808.863f);
    pts.emplace_back(929.145f, 744.976f);
    pts.emplace_back(917.474f, 791.823f);

    BOOST_CHECK(fit_and_check_ellipse(pts));
}

BOOST_AUTO_TEST_CASE(test_float2)
{
    std::vector<Point3f> pts;
    pts.reserve(10);
    pts.emplace_back(924.784f, 764.160f);
    pts.emplace_back(928.388f, 615.903f);
    pts.emplace_back(847.400f, 888.014f);
    pts.emplace_back(929.406f, 741.675f);
    pts.emplace_back(904.564f, 825.605f);
    pts.emplace_back(926.742f, 760.746f);
    pts.emplace_back(863.479f, 873.406f);
    pts.emplace_back(910.987f, 808.863f);
    pts.emplace_back(929.145f, 744.976f);
    pts.emplace_back(917.474f, 791.823f);

    BOOST_CHECK(fit_and_check_ellipse(pts));
}

BOOST_AUTO_TEST_CASE(test_withGT)
{
    // const float aGT = 5.0f;
    // const float bGT = 3.0f;
    // some points satisfying x^2/aGT^2 + y^2/bGT^2 = 1
    std::vector<Point3f> pts;
    pts.reserve(21);
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

    BOOST_CHECK(fit_and_check_ellipse(pts));

    cctag::numerical::geometry::Ellipse ellipse;
    cctag::numerical::ellipseFitting(ellipse, pts);

}

BOOST_AUTO_TEST_CASE(test_throw_5points)
{
    // test with 0 to 4 input point, it should throw
    for(std::size_t i = 0; i < 5; ++i)
    {
        std::vector<Point3f> pts;
        pts.reserve(i);
        for(std::size_t j = 0; j < i; ++j)
        {
            pts.emplace_back( 5.0000f, .0f);
        }
        cctag::numerical::geometry::Ellipse ellipse;

        BOOST_CHECK_THROW(cctag::numerical::ellipseFitting(ellipse, pts), std::domain_error);

    }
}

BOOST_AUTO_TEST_CASE(test_throw_degenerate_same_point)
{
    // test with 5 times the same point, degenerate case, it should throw
    std::vector<Point3f> pts;
    pts.reserve(5);
    for (int j = 0; j < 5; ++j)
    {
        pts.emplace_back( 5.0000f, .0f);
    }
    cctag::numerical::geometry::Ellipse ellipse;
    BOOST_REQUIRE_THROW(cctag::numerical::ellipseFitting(ellipse, pts), std::domain_error);
}

BOOST_AUTO_TEST_CASE(test_throw_degenerate_aligned_points)
{
    // test with 5 points on a line, it should throw
    std::vector<Point3f> pts;
    pts.reserve(5);
    pts.emplace_back( 5.0000f, 1.0f);
    pts.emplace_back( 5.0000f, 2.0f);
    pts.emplace_back( 5.0000f, 3.0f);
    pts.emplace_back( 5.0000f, 4.0f);
    pts.emplace_back( 5.0000f, 5.0f);
    cctag::numerical::geometry::Ellipse ellipse;
    BOOST_REQUIRE_THROW(cctag::numerical::ellipseFitting(ellipse, pts), std::domain_error);
}

BOOST_AUTO_TEST_CASE(test_throw_repeated_points)
{
    // test with 3 points and 2 of them repeated twice, it should throw
    std::vector<Point3f> pts;
    pts.reserve(5);
    pts.emplace_back( 0.0000f, 1.0f);
    pts.emplace_back( 0.0000f, 1.0f);
    pts.emplace_back( 0.0000f, 0.0f);
    pts.emplace_back( 1.0000f, 0.0f);
    pts.emplace_back( 1.0000f, 0.0f);
    cctag::numerical::geometry::Ellipse ellipse;
//    cctag::numerical::ellipseFitting(ellipse, pts);
//    std::cout << "a " << ellipse.a() << std::endl;
//    std::cout << "b " << ellipse.b() << std::endl;
//    std::cout << "angle " << ellipse.angle() << std::endl;
//    std::cout << "center " << ellipse.center() << std::endl;
//    const auto& m = ellipse.matrix();
//    std::cout << ellipse<< std::endl;
//    std::cout << m<< std::endl;
//    std::cout << m.topLeftCorner<2,2>().determinant() << std::endl;
//    std::cout << m.determinant() << std::endl;
//    std::cout << (m(0,0)+m(1,1))*m.determinant() << std::endl;
//
//    for (const auto& p : pts)
//    {
//        std::cout << p.transpose() * m * p<< std::endl;
//    }

// TODO: This test is disabled for now as Eigen computeInverseWithCheck in "fit_solver"
//       says that the matrix is invertible in this case and it should not be the case.
//       See here: https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html#title23
//     BOOST_REQUIRE_THROW(cctag::numerical::ellipseFitting(ellipse, pts), std::domain_error);
}

BOOST_AUTO_TEST_CASE(test_throw_degenerate_points)
{
    // test with degenerate configuration of points, must throw
    std::vector<Point3f> pts;
    pts.reserve(5);
    pts.emplace_back( -0.636353f,  5.27272f);
    pts.emplace_back(  0.363647f,  4.27272f);
    pts.emplace_back(  0.363647f,  1.27272f);
    pts.emplace_back(  0.363647f,  2.27272f);
    pts.emplace_back(  0.363647f,  3.27272f);
    pts.emplace_back(  0.363647f,  0.27272f);
    pts.emplace_back(  0.363647f, -0.72728f);
    pts.emplace_back(  0.363647f, -1.72728f);
    pts.emplace_back( -0.636353f, -3.72728f);
    pts.emplace_back( -0.636353f, -5.72728f);
    pts.emplace_back( -0.636353f, -4.72728f);

    cctag::numerical::geometry::Ellipse ellipse;
    BOOST_REQUIRE_THROW(cctag::numerical::ellipseFitting(ellipse, pts), std::domain_error);
}

BOOST_AUTO_TEST_SUITE_END()

