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
#include <cctag/Plane.hpp>
#include <cctag/CCTagMarkersBank.hpp>

#include <boost/ptr_container/ptr_list.hpp>

namespace cctag {

namespace logtime {
struct Mgmt;
}
  
using MarkerID = int;

class ICCTag
{
public:

    ICCTag()
        : _x( 0.f )
        , _y( 0.f )
        , _id( -1 )
    { }
                
	virtual float x() const = 0;
	virtual float y() const = 0;
	virtual MarkerID id() const = 0;
	virtual int getStatus() const = 0;

    virtual ~ICCTag() = default;

    virtual ICCTag* clone() const = 0;


protected:
    float _x;
    float _y;
    MarkerID _id;
        int _status; // WARNING: only markers with status == 1 are the valid ones. (status available via getStatus()) 
                     // A marker correctly detected and identified has a status 1.
                     // Otherwise, it can be detected but not correctly identified.
};

inline ICCTag* new_clone(const ICCTag& a)
{
    return a.clone();
}

/**
 * @brief Perform the CCTag detection on a gray scale image
 * 
 * @param[out] markers Detected markers. WARNING: only markers with status == 1 are the valid ones. (status available via getStatus()) 
 * @param[in] frame A frame number. Can be anything (e.g. 0).
 * @param[in] graySrc Gray scale input image.
 * @param[in] nRings Number of CCTag rings.
 * @param[in] parameterFile Path to a parameter file. If not provided default parameters will be used.
 * @param[in] cctagBankFilename Path to the cctag bank. If not provided, radii will be the ones associated to the CCTags contained in the
 * markersToPrint folder.
 */
void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      int                       pipeId,
      std::size_t frame,
      Plane<uint8_t>& graySrc,
      std::size_t nRings = 3,
      logtime::Mgmt* durations = nullptr,
      const std::string & parameterFile = "",
      const std::string & cctagBankFilename = "");

void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      int                       pipeId,
      std::size_t frame,
      Plane<uint8_t>& graySrc,
      const cctag::Parameters & params,
      logtime::Mgmt* durations = nullptr,
      const CCTagMarkersBank * pBank = nullptr);

}

#endif	/* PONCTUALCCTAG_HPP */

