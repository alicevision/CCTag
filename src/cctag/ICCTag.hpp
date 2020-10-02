/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef PONCTUALCCTAG_HPP
#define	PONCTUALCCTAG_HPP

#include <cctag/Params.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <boost/ptr_container/ptr_list.hpp>

#include <opencv2/core/core.hpp>

namespace cctag {

namespace logtime {
struct Mgmt;
}

using MarkerID = int;
static constexpr MarkerID UndefinedMarkerID{-1};

class ICCTag
{
public:
    /**
     * @brief Default constructor
     */
    ICCTag() = default;

    /**
     * @brief Ger the x coordinate of the center of the marker.
     * @return x coordinate of the center.
     */
    virtual float x() const = 0;

    /**
     * @brief Ger the x coordinate of the center of the marker.
     * @return x coordinate of the center.
     */
    virtual float y() const = 0;

    /**
     * @brief Get marker ID.
     * @return the marker ID.
     */
    virtual MarkerID id() const = 0;

    /**
     * @brief Get the status of the marker.
     * @return the status of the marker.
     */
    virtual int getStatus() const = 0;
    /**
     * @brief Get the rescaled outer ellipse of the marker.
     * The rescaled outerEllipse is in the coordinate system of the input image, while the internal ellipse is
     * relative to a pyramid level.
     * @return the outer ellipse.
     */
    virtual const cctag::numerical::geometry::Ellipse & rescaledOuterEllipse() const = 0;

    virtual ~ICCTag() = default;

    virtual ICCTag* clone() const = 0;


protected:
    /// x coordinate of the center of the marker
    float _x{0.f};
    /// y coordinate of the center of the marker
    float _y{0.f};
    /// numeric ID of the marker
    MarkerID _id{UndefinedMarkerID};
    /// Status of the marker: only markers with status equal tp 1 are the correctly detected and identified. (@see getStatus())
    /// Otherwise, it can be detected but not correctly identified.
    int _status{-1};
};

inline ICCTag* new_clone(const ICCTag& a)
{
    return a.clone();
}

/**
 * @brief Perform the CCTag detection on a gray scale image
 *
 * @param[out] markers Detected markers. WARNING: only markers with status == 1 are valid ones. (status available via
 * getStatus())
 * @param[in] pipeId Choose between several CUDA pipeline instances
 * @param[in] frame A frame number. Can be anything (e.g. 0).
 * @param[in] graySrc Gray scale input image.
 * @param[in] nRings Number of CCTag rings.
 * @param[in] durations Optional object to store execution times.
 * @param[in] parameterFilename Path to a parameter file. If not provided default parameters will be used.
 * @param[in] cctagBankFilename Path to the cctag bank. If not provided, radii will be the ones associated to the CCTags
 * contained in the markersToPrint folder.
 */
void cctagDetection(boost::ptr_list<ICCTag>& markers,
                    int pipeId,
                    std::size_t frame,
                    const cv::Mat& graySrc,
                    std::size_t nRings = 3,
                    logtime::Mgmt* durations = nullptr,
                    const std::string& parameterFile = "",
                    const std::string& cctagBankFilename = "");

/**
 * @brief Perform the CCTag detection on a gray scale image
 *
 * @param[out] markers Detected markers. WARNING: only markers with status == 1 are valid ones. (status available via
 * getStatus())
 * @param[in] pipeId Choose between several CUDA pipeline instances
 * @param[in] frame A frame number. Can be anything (e.g. 0).
 * @param[in] graySrc Gray scale input image.
 * @param[in] params Parameters for the detection.
 * @param[in] durations Optional object to store execution times.
 * @param[in] pBank Path to the cctag bank. If not provided, radii will be the ones associated to the CCTags contained
 * in the markersToPrint folder.
 */
void cctagDetection(boost::ptr_list<ICCTag>& markers,
                    int pipeId,
                    std::size_t frame,
                    const cv::Mat& graySrc,
                    const cctag::Parameters& params,
                    logtime::Mgmt* durations = nullptr,
                    const CCTagMarkersBank* pBank = nullptr);

}

#endif	/* PONCTUALCCTAG_HPP */

