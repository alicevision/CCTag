/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_MARKER_CCTAG_CCTAG_HPP
#define VISION_MARKER_CCTAG_CCTAG_HPP

#include <cctag/Types.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/Candidate.hpp>
#include <cctag/CCTagFlowComponent.hpp>
#include <cctag/geometry/Point.hpp>
// #include <cctag/algebra/Invert.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/Types.hpp>
#include <cctag/ICCTag.hpp>
#include <cctag/geometry/2DTransform.hpp>
#include <cctag/utils/Defines.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/array.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/throw_exception.hpp>
#include <boost/archive/text_oarchive.hpp>

#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/pinned_counters.h"
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace cctag
{

using IdSet = std::vector< std::pair< MarkerID, float >>;

/**
 * @brief Class modeling the CCTag marker containing the position of the marker in the image, its ID and its status.
 */
class CCTag : public ICCTag
{
public:
  using Vector = boost::ptr_vector<CCTag>;
  using List = boost::ptr_list<CCTag>;

public:

  CCTag()
    : _id(0)
    , _quality(0)
    , _status(0)
#ifdef CCTAG_WITH_CUDA
    , _cuda_result( nullptr )
#endif
  {
    setInitRadius();
  }

  CCTag(const MarkerID id,
        const cctag::Point2d<Eigen::Vector3f> & centerImg,
        const std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > > & points,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const Eigen::Matrix3f & homography,
        int pyramidLevel,
        float scale,
        const float quality = 1.f)
    : _centerImg(centerImg)
    , _id(id)
    , _outerEllipse(outerEllipse)
    , _points(points)
    , _mHomography(homography)
    , _quality(quality)
    , _pyramidLevel(pyramidLevel)
    , _scale(scale)
#ifdef CCTAG_WITH_CUDA
    , _cuda_result( nullptr )
#endif
  {
    setInitRadius();
    _outerEllipse.setCenter( Point2d<Eigen::Vector3f>(_outerEllipse.center().x()+0.5f, _outerEllipse.center().y()+0.5f ) ); // todo@Lilian: + 0.5f
    cctag::numerical::geometry::scale(_outerEllipse, _rescaledOuterEllipse, scale);
    
    _status = 0;
  }

  CCTag(const CCTag & cctag)
    : _centerImg(cctag._centerImg)
    , _nCircles(cctag._nCircles)
    , _radiusRatios(cctag._radiusRatios)
    , _id(cctag._id)
    , _outerEllipse(cctag._outerEllipse)
    , _ellipses(cctag._ellipses)
    , _points(cctag._points)
    , _mHomography(cctag._mHomography)
    , _quality(cctag._quality)
    , _pyramidLevel(cctag._pyramidLevel)
    , _scale(cctag._scale)
    , _rescaledOuterEllipse(cctag._rescaledOuterEllipse)
    , _status(cctag._status)
#ifdef CCTAG_WITH_CUDA
    , _cuda_result( nullptr )
#endif
#ifdef CCTAG_SERIALIZE
    , _flowComponents(cctag._flowComponents)
#endif
  {
  }

  ~CCTag() override = default;

#ifndef NDEBUG
  void printTag( std::ostream& ostr ) const;
#endif

  void applyScale(float s);

  float x() const override {
    return _centerImg.x();
  }
  
  float y() const override {
    return _centerImg.y();
  }
  
  std::size_t nCircles() const
  {
    return _nCircles;
  }

  const Eigen::Matrix3f & homography() const
  {
    return _mHomography;
  }
  
  Point2d<Eigen::Vector3f> & centerImg()
  {
    return _centerImg;
  }
  
  void setCenterImg( const cctag::Point2d<Eigen::Vector3f>& center )
  {
    _centerImg = center;
  }

  Eigen::Matrix3f & homography()
  {
    return _mHomography;
  }

  void setHomography(const Eigen::Matrix3f & homography)
  {
    _mHomography = homography;
  }

  const cctag::numerical::geometry::Ellipse & outerEllipse() const
  {
    return _outerEllipse;
  }

  const std::vector< DirectedPoint2d<Eigen::Vector3f> > & rescaledOuterEllipsePoints() const
  {
    return _rescaledOuterEllipsePoints;
  }

  const std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > >& points() const
  {
    return _points;
  }

  static const boost::array<float, 5> & radiusRatiosInit()
  {
    return _radiusRatiosInit;
  }

  const std::vector<float>& radiusRatios() const
  {
    return _radiusRatios;
  }

  std::vector<float> & radiusRatios()
  {
    return _radiusRatios;
  }

  void setRadiusRatios(const std::vector<float>& radiusRatios)
  {
    _radiusRatios = radiusRatios;
  }

  float quality() const
  {
    return _quality;
  }

  void setQuality(const float quality)
  {
    _quality = quality;
  }

  float scale() const
  {
    return _scale;
  }

  void setScale(const float scale)
  {
    _scale = scale;
  }

  int pyramidLevel() const
  {
    return _pyramidLevel;
  }

  void setPyramidLevel(const int pyramidLevel)
  {
    _pyramidLevel = pyramidLevel;
  }

  std::vector<cctag::numerical::geometry::Ellipse> & ellipses()
  {
    return _ellipses;
  }

  std::vector<cctag::numerical::geometry::Ellipse> ellipses() const
  {
    return _ellipses;
  }

  void setEllipses(const std::vector<cctag::numerical::geometry::Ellipse>& ellipses)
  {
    _ellipses = ellipses;
  }

  void setRescaledOuterEllipse(const cctag::numerical::geometry::Ellipse & rescaledOuterEllipse)
  {
    _rescaledOuterEllipse = rescaledOuterEllipse;
  }

  void setRescaledOuterEllipsePoints(const std::vector< DirectedPoint2d<Eigen::Vector3f> > & outerEllipsePoints)
  {
    _rescaledOuterEllipsePoints = outerEllipsePoints;
  }

  const cctag::numerical::geometry::Ellipse & rescaledOuterEllipse() const override
  {
    return _rescaledOuterEllipse;
  }

  bool hasId() const
  {
    return true;
  }

  MarkerID id() const override
  {
    return _id;
  }

  void setId(const MarkerID id)
  {
    _id = id;
  }

  IdSet idSet() const
  {
    return _idSet;
  }

  void setIdSet(const IdSet& idSet)
  {
    _idSet = idSet;
  }

  int getStatus() const override
  {
    return _status;
  }


  void setStatus(int status)
  {
    _status = status;
  }

  bool operator<(const CCTag & tag2) const
  {
    return _id < tag2.id();
  }

  //friend std::ostream& operator<<(std::ostream& os, const CCTag& cm);

  inline CCTag* clone() const override
  {
    return new CCTag(*this);
  }

  void condition(const Eigen::Matrix3f & mT, const Eigen::Matrix3f & mInvT);

  bool isOverlapping(const CCTag& marker) const
  {
    return isOverlappingEllipses(_rescaledOuterEllipse, marker.rescaledOuterEllipse());
  }
  
  bool isEqual(const CCTag& marker) const;

#ifdef CCTAG_SERIALIZE

  void addFlowComponent(const Candidate & candidate, const EdgePointCollection& edgeCollection)
  {
    _flowComponents.emplace_back(
        edgeCollection,
        candidate._outerEllipsePoints,
        candidate._children,
        candidate._filteredChildren,
        candidate._outerEllipse,
        candidate._convexEdgeSegment,
        *(candidate._seed),
        _nCircles);
  }

  void setFlowComponents(const std::vector<CCTagFlowComponent> & flowComponents)
  {
    _flowComponents = flowComponents;
  }

  void setFlowComponents(const std::vector<Candidate> & candidates, const EdgePointCollection& edgeCollection)
  {
    for(const Candidate & candidate : candidates)
    {
      addFlowComponent(candidate, edgeCollection);
    }
  }

  const std::vector<CCTagFlowComponent> & getFlowComponents() const
  {
    return _flowComponents;
  }
#endif

#ifdef CCTAG_WITH_CUDA
  /** Get a pointer to pinned memory for this tag.
   *  It cannot be released for this tag.
   *  Instead, releaseNearbyPointMemory() invalidates all such
   *  pointers in the process.
   */
  void acquireNearbyPointMemory( int pipeId );

  inline cctag::NearbyPoint* getNearbyPointBuffer( ) {
    return _cuda_result;
  }

  /** Release all pinned memory associated with NearbyPoints.
   *  Invalidates pointers in all objects and in all threads in
   *  this process.
   */
  static void releaseNearbyPointMemory( int pipeId );
#endif

  void serialize(boost::archive::text_oarchive & ar, unsigned int version);

protected:

  void setInitRadius()
  {
    _radiusRatios.resize(CCTag::_radiusRatiosInit.size());
    std::copy(CCTag::_radiusRatiosInit.begin(), CCTag::_radiusRatiosInit.end(), _radiusRatios.begin());
    _nCircles = CCTag::_radiusRatiosInit.size() + 1;
  }

protected:
  static const boost::array<float, 5> _radiusRatiosInit;
  std::vector<float> _radiusRatios;

  std::size_t _nCircles;
  MarkerID _id;
  IdSet _idSet;
  cctag::Point2d<Eigen::Vector3f> _centerImg;
  cctag::numerical::geometry::Ellipse _outerEllipse;
  cctag::numerical::geometry::Ellipse _rescaledOuterEllipse;
  std::vector< DirectedPoint2d<Eigen::Vector3f> > _rescaledOuterEllipsePoints;
  std::vector<cctag::numerical::geometry::Ellipse> _ellipses;
  std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > > _points;
  Eigen::Matrix3f _mHomography;
  float _quality;
  int    _pyramidLevel;
  float _scale;
  int    _status;
#ifdef CCTAG_WITH_CUDA
  /** Pointer into pinned memory page.
   *  Valid from the construction of the CCTag until identify()
   *  is complete.
   */
  cctag::NearbyPoint* _cuda_result;
#endif

#ifdef CCTAG_SERIALIZE
  std::vector<CCTagFlowComponent> _flowComponents;
#endif
};

inline CCTag* new_clone(const CCTag& x)
{
  return x.clone();
}

namespace status
{
// List of possible status
static const int id_reliable = 1;
static const int too_few_outer_points = -1;
static const int no_collected_cuts = -1;
static const int no_selected_cuts = -2;
static const int opti_has_diverged = -3;
static const int id_not_reliable = -4;
static const int degenerate = -5;
}

} // namespace cctag

#endif
