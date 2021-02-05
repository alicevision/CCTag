/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Multiresolution.hpp>
#include <cctag/utils/VisualDebug.hpp>
#include <cctag/utils/FileDebug.hpp>
#include <cctag/Vote.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/Canny.hpp>
#include <cctag/Detection.hpp>
#include <cctag/utils/Talk.hpp> // for DO_TALK macro

#include <boost/timer/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <sstream>
#include <fstream>
#include <map>

#include <limits>

#ifdef CCTAG_WITH_CUDA
#include <cctag/cuda/cctag_cuda_runtime.h> // only for debugging!!!
#include "cctag/cuda/tag.h"
#endif


namespace cctag
{

/* @brief Add markers from a list to another, deleting duplicates.
 *
 *
 */

static bool intersectLineToTwoEllipses(
        std::ssize_t y,
        const numerical::geometry::Ellipse & qIn,
        const numerical::geometry::Ellipse & qOut,
        const EdgePointCollection & edgeCollection,
        std::list<EdgePoint*> & pointsInHull)
{
  std::vector<float> intersectionsOut = numerical::geometry::intersectEllipseWithLine(qOut, y, true);
  std::vector<float> intersectionsIn = numerical::geometry::intersectEllipseWithLine(qIn, y, true);
  BOOST_ASSERT(intersectionsOut.size() <= 2);
  BOOST_ASSERT(intersectionsIn.size() <= 2);
  if ((intersectionsOut.size() == 2) && (intersectionsIn.size() == 2))
  {
    std::ssize_t begin1 = std::max(0, (int) intersectionsOut[0]);
    std::ssize_t end1 = std::min((int) edgeCollection.shape()[0] - 1, (int) intersectionsIn[0]);

    std::ssize_t begin2 = std::max(0, (int) intersectionsIn[1]);
    std::ssize_t end2 = std::min((int) edgeCollection.shape()[0] - 1, (int) intersectionsOut[1]);

    for (int x = begin1; x <= end1; ++x)
    {
      EdgePoint* edgePoint = edgeCollection(x,y);
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        Eigen::Vector2f centerToPoint;
        centerToPoint(0) = qIn.center().x() - (*edgePoint).x();
        centerToPoint(1) = qIn.center().y() - (*edgePoint).y();
        
        if (edgePoint->gradient().dot(centerToPoint) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
    for (int x = begin2; x <= end2; ++x)
    {
      EdgePoint* edgePoint = edgeCollection(x,y);
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        Eigen::Vector2f centerToPoint;
        centerToPoint(0) = qIn.center().x() - (*edgePoint).x();
        centerToPoint(1) = qIn.center().y() - (*edgePoint).y();
        
        if (edgePoint->gradient().dot(centerToPoint) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
  }
  else if ((intersectionsOut.size() == 2) && (intersectionsIn.size() <= 1))
  {
    std::ssize_t begin = std::max(0, (int) intersectionsOut[0]);
    std::ssize_t end = std::min((int) edgeCollection.shape()[0] - 1, (int) intersectionsOut[1]);

    for (int x = begin; x <= end; ++x)
    {
      EdgePoint* edgePoint = edgeCollection(x,y);
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        
        Eigen::Vector2f centerToPoint;
        centerToPoint(0) = qIn.center().x() - (*edgePoint).x();
        centerToPoint(1) = qIn.center().y() - (*edgePoint).y();
        
        if (edgePoint->gradient().dot(centerToPoint) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
  }
  else if ((intersectionsOut.size() == 1) && (intersectionsIn.size() == 0))
  {
    if ((intersectionsOut[0] >= 0) && (intersectionsOut[0] < edgeCollection.shape()[0]))
    {
      EdgePoint* edgePoint = edgeCollection(intersectionsOut[0],y);
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        Eigen::Vector2f centerToPoint;
        centerToPoint(0) = qIn.center().x() - (*edgePoint).x();
        centerToPoint(1) = qIn.center().y() - (*edgePoint).y();
        
        if (edgePoint->gradient().dot(centerToPoint) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
  }
  else //if( intersections.size() == 0 )
  {
    return false;
  }
  return true;
}

static void selectEdgePointInEllipticHull(
        const EdgePointCollection & edgeCollection,
        const numerical::geometry::Ellipse & outerEllipse,
        float scale,
        std::list<EdgePoint*> & pointsInHull)
{
  numerical::geometry::Ellipse qIn, qOut;
  computeHull(outerEllipse, scale, qIn, qOut);

  const float yCenter = outerEllipse.center().y();

  int maxY = std::max(int(yCenter), 0);
  int minY = std::min(int(yCenter), int(edgeCollection.shape()[1]) - 1);

  // Visit the bottom part of the ellipse
  for (std::ssize_t y = maxY; y < int( edgeCollection.shape()[1]); ++y)
  {
    if (!intersectLineToTwoEllipses(y, qIn, qOut, edgeCollection, pointsInHull))
      break;
  }
  // Visit the upper part of the ellipse
  for (std::ssize_t y = minY; y >= 0; --y)
  {
    if (!intersectLineToTwoEllipses(y, qIn, qOut, edgeCollection, pointsInHull))
      break;
  }
}

void update(
        CCTag::List& markers,
        const CCTag& markerToAdd)
{
  bool flag = false;

  for(CCTag & currentMarker : markers)
  {
    if ( ( currentMarker.getStatus() > 0 ) && ( markerToAdd.getStatus() > 0 ) && currentMarker.isEqual(markerToAdd) )
    {
      if (markerToAdd.quality() > currentMarker.quality())
      {
        currentMarker = markerToAdd;
      }
      flag = true;
    }
  }
  if (!flag)
  {
    markers.push_back(new CCTag(markerToAdd));
  }
}

static void cctagMultiresDetection_inner(
        size_t                  i,
        CCTag::List&            pyramidMarkers,
        const cv::Mat&          imgGraySrc,
        Level*                  level,
        const std::size_t       frame,
        EdgePointCollection&    edgeCollection,
        cctag::TagPipe*        cuda_pipe,
        const Parameters &      params,
        cctag::logtime::Mgmt*   durations )
{
    DO_TALK( CCTAG_COUT_OPTIM(":::::::: Multiresolution level " << i << "::::::::"); )

    // Data structure for getting vote winners
    std::vector<EdgePoint*> seeds;

    boost::posix_time::time_duration d;

#if defined(CCTAG_WITH_CUDA)
    // there is no point in measuring time in compare mode
    if( cuda_pipe ) {
      cuda_pipe->convertToHost(i, edgeCollection, seeds, cctag::EdgePointCollection::MAX_POINTS );
      if( durations ) {
          cudaDeviceSynchronize();
      }
      level->setLevel( cuda_pipe, params );

      CCTagVisualDebug::instance().setPyramidLevel(i);
    } else { // not cuda_pipe
#endif // defined(CCTAG_WITH_CUDA)
    edgesPointsFromCanny( edgeCollection,
                          level->getEdges(),
                          level->getDx(),
                          level->getDy());

    CCTagVisualDebug::instance().setPyramidLevel(i);

    // Voting procedure applied on every edge points.
    vote( edgeCollection,
          seeds,        // output
          level->getDx(),
          level->getDy(),
          params );
    
    if( seeds.size() > 1 ) {
        // Sort the seeds based on the number of received votes.
        std::sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);
    }

#if defined(CCTAG_WITH_CUDA)
    } // not cuda_pipe
#endif // defined(CCTAG_WITH_CUDA)


    cctagDetectionFromEdges(
        pyramidMarkers,
        edgeCollection,
        level->getSrc(),
        seeds,
        frame, i, std::pow(2.0, (int) i), params,
        durations );

    CCTagVisualDebug::instance().initBackgroundImage(level->getSrc());
    std::stringstream outFilename2;
    outFilename2 << "viewLevel" << i;
    CCTagVisualDebug::instance().newSession(outFilename2.str());

    for(const CCTag & marker : pyramidMarkers)
    {
        CCTagVisualDebug::instance().drawMarker(marker, false);
    }
}

void cctagMultiresDetection(
        CCTag::List& markers,
        const cv::Mat& imgGraySrc,
        const ImagePyramid& imagePyramid,
        std::size_t   frame,
        cctag::TagPipe*    cuda_pipe,
        const Parameters&   params,
        cctag::logtime::Mgmt* durations )
{
  //	* For each pyramid level:
  //	** launch CCTag detection based on the canny edge detection output.

  // std::map<std::size_t, CCTag::List> pyramidMarkers;
  const int numProcLayers = params._numberOfProcessedMultiresLayers;

  std::vector<std::unique_ptr<EdgePointCollection> > vEdgePointCollections( numProcLayers );
  for( int i = 0; i<numProcLayers; i++ )
  {
    vEdgePointCollections[i].reset(new EdgePointCollection(imgGraySrc.cols, imgGraySrc.rows));
  }

  BOOST_ASSERT( params._numberOfMultiresLayers - numProcLayers >= 0 );
  for( int i = numProcLayers-1; i >= 0; i-- )
  {
    CCTag::List pyramidMarkers;
    
    cctagMultiresDetection_inner( i,
                                  pyramidMarkers,
                                  imgGraySrc,
                                  imagePyramid.getLevel(i),
                                  frame,
                                  *vEdgePointCollections[i],
                                  cuda_pipe,
                                  params,
                                  durations );

    // Gather the detected markers in the entire image pyramid
    for(const CCTag & marker : pyramidMarkers)
    {
      markers.push_back( new CCTag(marker) );
    }
  }
  if( durations ) durations->log( "after cctagMultiresDetection_inner" );
  
  CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(0)->getSrc());
  CCTagVisualDebug::instance().writeLocalizationView(markers);

  // Final step: extraction of the detected markers in the original (scale) image.
  CCTagVisualDebug::instance().newSession("multiresolution");

  // Project markers from the top of the pyramid to the bottom (original image).
  for(CCTag & marker : markers)
  {
    int i = marker.pyramidLevel();
    // if the marker has to be rescaled into the original image
    if (i > 0)
    {
      BOOST_ASSERT( i < params._numberOfMultiresLayers );
      float scale = marker.scale(); // pow( 2.0, (float)i );

      cctag::numerical::geometry::Ellipse rescaledOuterEllipse = marker.rescaledOuterEllipse();

      #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
      #endif
      
      
      std::list<EdgePoint*> pointsInHull;
      selectEdgePointInEllipticHull( *vEdgePointCollections[0],
                                     rescaledOuterEllipse,
                                     scale,
                                     pointsInHull );

      #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
      #endif
      

      if(pointsInHull.size() < 5)
          continue;

      std::vector<EdgePoint*> rescaledOuterEllipsePoints;

      float SmFinal = 1e+10;
      
      cctag::outlierRemoval(
              pointsInHull,
              rescaledOuterEllipsePoints,
              SmFinal,
              20.0,
              NO_WEIGHT,
              60); 
      
      #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
        boost::posix_time::time_duration d1 = t1 - t0;
        boost::posix_time::time_duration d2 = t2 - t1;
        CCTAG_COUT_OPTIM("Time in selectEdgePointInEllipticHull: " << d1.total_milliseconds() << " ms");
        CCTAG_COUT_OPTIM("Time in outlierRemoval: " << d2.total_milliseconds() << " ms");
      #endif

      if(rescaledOuterEllipsePoints.size() < 5)
        continue;

      try
      {
        numerical::ellipseFitting(rescaledOuterEllipse, rescaledOuterEllipsePoints);

        std::vector< DirectedPoint2d<Eigen::Vector3f> > rescaledOuterEllipsePointsDouble;
        std::size_t numCircles = params._nCrowns * 2;

        for(EdgePoint * e : rescaledOuterEllipsePoints)
        {
          rescaledOuterEllipsePointsDouble.emplace_back(e->x(), e->y(),
                  e->dX(),
                  e->dY()
          );
          
          CCTagVisualDebug::instance().drawPoint(Point2d<Eigen::Vector3f>(e->x(), e->y()), cctag::color_red);
        }
        marker.setCenterImg(cctag::Point2d<Eigen::Vector3f>(marker.centerImg().x() * scale, marker.centerImg().y() * scale));
        marker.setRescaledOuterEllipse(rescaledOuterEllipse);
        marker.setRescaledOuterEllipsePoints(rescaledOuterEllipsePointsDouble);
      }
      catch (...)
      {
        // catch exception from ellipseFitting
      }
    }
    else
    {
      marker.setRescaledOuterEllipsePoints(marker.points().back());
    }
  }
  if( durations ) durations->log( "after marker projection" );

  // Log
  CCTagFileDebug::instance().newSession("data.txt");
  for(const CCTag & marker : markers)
  {
    CCTagFileDebug::instance().outputMarkerInfos(marker);
  }
  
  // POP_LEAVE;
  
}

} // namespace cctag

