/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_NUMERICAL_ELLIPSE_HPP_
#define _CCTAG_NUMERICAL_ELLIPSE_HPP_

#include <cctag/geometry/Point.hpp>
#include <Eigen/Core>
#include <iostream>

namespace cctag {
namespace numerical {
namespace geometry {

/**
 * @brief It models an ellipse with standard form \f$ \frac{x^2 - x_c}{a^2} + \frac{y^2 - y_c}{b^2} = 1 \f$, centered in
 * \p _center \f$(x_c, x_y)\f$ and rotated clock-wise by \p _angle wrt the x-axis. Note that, arbitrarly, the
 * representation with the major axis aligned with the y-axis is chosen.
 */
class Ellipse
{
public:
    using Matrix = Eigen::Matrix3f;

    /**
     * @brief Default constructor, set all parameters to zero.
     */
    Ellipse() = default;

    /**
     * @brief Build an ellipse from a 3x3 matrix representing the ellipse as a conic.
     * @param[in] matrix The 3x3 matrix representing the ellipse.
     * @note By default, the representation with the major axis aligned with the y-axis is chosen.
     */
    explicit Ellipse(const Matrix& matrix);

    /**
     * @brief Build an ellipse from a set of parameters.
     * @param[in] center The center of the conic.
     * @param[in] a The length of the semi-axis x.
     * @param[in] b The length of the semi-axis y.
     * @param[in] angle The orientation of the ellipse wrt the x-axis as a clock-wise angle in radians.
     */
    Ellipse(const Point2d <Eigen::Vector3f>& center, float a, float b, float angle);

    /**
     * @brief Return the matrix representation of the ellipse.
     * @return 3x3 matrix representation of the ellipse.
     */
    inline const Matrix &matrix() const { return _matrix; }

    /**
     * @brief Return the matrix representation of the ellipse.
     * @return 3x3 matrix representation of the ellipse.
     */
    inline Matrix& matrix() { return _matrix; }

    /**
     * @brief Return the center of the ellipse.
     * @return 3 element vector with the homogeneous coordinates of the ellipse.
     */
    inline const Point2d <Eigen::Vector3f> &center() const { return _center; }

    /**
     * @brief Return the center of the ellipse.
     * @return 3 element vector with the homogeneous coordinates of the ellipse.
     */
    inline Point2d <Eigen::Vector3f> &center() { return _center; }

    /**
     * @brief Return the length of the x-semi axis of the ellipse.
     * @return the length of the x-semi axis of the ellipse.
     */
    inline float a() const { return _a; }

    /**
     * @brief Return the length of the y-semi axis of the ellipse.
     * @return the length of the y-semi axis of the ellipse.
     */
    inline float b() const { return _b; }

    /**
     * @brief Return the orientation of the ellipse.
     * @return the clock-wise orientation angle in radians of the ellipse wrt the x-axis
     */
    inline float angle() const { return _angle; }

    /**
     * @brief Set the length of the x-semi axis of the ellipse.
     * @param[in] a the length of the x-semi axis.
     */
    void setA(float a);

    /**
     * @brief Set the length of the y-semi axis of the ellipse.
     * @param[in] b the length of the y-semi axis.
     */
    void setB(float b);

    /**
     * @brief Set the orientation angle of the ellipse.
     * @param[in] angle the clock-wise orientation angle in radians.
     */
    void setAngle(float angle);

    /**
     * @brief Set the center of the ellipse.
     * @param[in] center the new center of the ellipse.
     */
    void setCenter(const Point2d <Eigen::Vector3f>& center);

    /**
     * @brief Update the ellipse from a matrix representing a conic.
     * @param[in] matrix 3x3 matric representing the ellipse.
     */
    void setMatrix(const Matrix& matrix);

    /**
     * @brief Update the ellipse from its parameters.
     * @param[in] center The center of the conic.
     * @param[in] a The length of the semi-axis x.
     * @param[in] b The length of the semi-axis y.
     * @param[in] angle The orientation of the ellipse wrt the x-axis as a clock-wise angle in radians.
     */
    void setParameters(const Point2d <Eigen::Vector3f>& center, float a, float b, float angle);

    /**
     * @brief Return a new ellipse obtained by applying a transformation to the ellipse.
     * @param[in] mT a 3x3 matrix representing the transformation.
     * @return the transformed ellipse.
     */
    Ellipse transform(const Matrix& mT) const;

    /**
     * @brief Compute the canonical form of the conic, along with its transformation.
     * @param[out] mCanonic 3x3 diagonal matrix representing the ellipse in canonical form.
     * @param[out] mTprimal 3x3 transformation matrix such that C = \p mTprimal.transpose() * \p mCanonic * \p mTprimal
     * @param[out] mTdual 3x3 inverse transformation matrix (= \p mTprimal.inv())
     */
    void getCanonicForm(Matrix& mCanonic, Matrix& mTprimal, Matrix& mTdual) const;

    /**
     * @brief Print the ellipse in matrix form in Matlab notation.
     * @param[in,out] os the stream where to output the ellipse.
     * @param[in] e the ellipse
     * @return the stream with appended the matrix representation of the ellipse.
     */
    friend std::ostream& operator<<(std::ostream& os, const Ellipse&e);

private:

    /**
     * @brief Starting from the matrix representation, it recomputes the parameters of the ellipse.
     */
    void computeParameters();

    /**
     * @brief Starting from the parameters, it recomputes the matrix representation of the ellipse.
     */
    void computeMatrix();

    /**
     * @brief Build an ellipse from a set of parameters.
     * @param[in] center The center of the conic.
     * @param[in] a The length of the semi-axis x.
     * @param[in] b The length of the semi-axis y.
     * @param[in] angle The orientation of the ellipse wrt the x-axis as a clock-wise angle in radians.
     */
    void init(const Point2d <Eigen::Vector3f> &center, float a, float b, float angle);

protected:
    /// the 3x3 matrix representation of the ellipse
    Eigen::Matrix3f _matrix{Eigen::Matrix3f::Zero()};
    /// the center of the ellipse
    Point2d <Eigen::Vector3f> _center{0, 0};
    /// the length of the major semi-axis
    float _a{0.f};
    /// the length of the minor semi-axis
    float _b{0.f};
    /// the angle between the major semi-axis and the x-axis
    float _angle{.0f};
};

/*
 * @brief Sort a set of points by angle along an elliptical arc. Possibly return a subset of these
 *        points if requested.
 * @param[in] ellipse the ellipse.
 * @param[in] points  the unordered points on the ellipse.
 * @param[out] resPoints  the ordered points set with size = min(requestedSize, points.size()).
 * @param[in] requestedSize the number to ordered points to return, the points are uniformely sampled along the list.
 */
void getSortedOuterPoints(
        const Ellipse& ellipse,
        const std::vector<cctag::DirectedPoint2d<Eigen::Vector3f>>& points,
        std::vector<cctag::DirectedPoint2d<Eigen::Vector3f>>& resPoints,
        std::size_t requestedSize);

/**
 * @brief Computes a new ellipse scaled by a given factor.
 * @param[in] ellipse the original ellipse.
 * @param[out] rescaleEllipse  the rescaled ellipse.
 * @param[in] scale the scale factor to apply.
 */
void scale(const Ellipse &ellipse, Ellipse &rescaleEllipse, float scale);

}
}
}

#endif

