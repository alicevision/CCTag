/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_CCTAG_VISUALDEBUG_HPP_
#define	_CCTAG_CCTAG_VISUALDEBUG_HPP_

#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/utils/Singleton.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/Colors.hpp>
#include <cctag/CCTag.hpp>

#include <boost/filesystem.hpp>

namespace cctag
{

class CCTagVisualDebug : public Singleton<CCTagVisualDebug> {
    MAKE_SINGLETON_WITHCONSTRUCTORS(CCTagVisualDebug)

public:
    using Sessions = std::map<std::string, cv::Mat>;
public:

    void setPyramidLevel(int level);

    int getPyramidLevel() const;
    
    void resetMarkerIndex();

#ifdef CCTAG_SERIALIZE
    void incrementMarkerIndex();
    
    std::size_t getMarkerIndex() const; 
#endif
    
    std::string getPath() const;

    void setImageFileName(const std::string& imageFileName);

    void initBackgroundImage(const cv::Mat & back);
    
    void initializeFolders(const boost::filesystem::path & rootPath, const std::string & outputFolder, std::size_t nCrowns = 4);

    void newSession(const std::string & sessionName);

    void drawText(const cctag::Point2d<Eigen::Vector3f> & p, const std::string & text, const cctag::Color & color);

    void drawPoint(const cctag::Point2d<Eigen::Vector3f> & point, const cctag::Color & color);
    void drawPoint(const cctag::DirectedPoint2d<Eigen::Vector3f> & point, const cctag::Color & color);

    void drawPoints(const std::vector<cctag::Point2d<Eigen::Vector3f> > & points, const cctag::Color & color);
    
    void drawPoints(const std::vector<cctag::DirectedPoint2d<Eigen::Vector3f> > & points, const cctag::Color & color);

    void drawMarker(const cctag::CCTag& marker, bool drawScaledMarker = true);

    void drawInfos(const cctag::CCTag& marker, bool drawScaledMarker = true);

    void out(const std::string & filename) const;

    void outPutAllSessions() const;

    void writeLocalizationView(cctag::CCTag::List & markers) const;

    void writeIdentificationView(cctag::CCTag::List & markers) const;

    std::string getImageFileName() const;
    
#ifdef CCTAG_EXTRA_LAYER_DEBUG
    template<typename T>
    void coutImage(cv::Mat src) const
    {
      for (int i=0 ; i < src.rows ; ++i)
      {
        for (int j=0 ; j < src.cols ; ++j)
        {
          std::cout << (int) src.at<T>(i,j) << " ";
        }
        std::cout << std::endl;
      }
    }
#endif

    void clearSessions();

private:
    Sessions _sessions; ///< Sessions map

    cv::Mat _backImage; // Background image
    int _pyramidLevel;
    std::string _imageFileName;
    std::string _pathRoot;
    std::string _path;
    std::size_t _markerIndex;
};

} // namespace cctag


#endif

