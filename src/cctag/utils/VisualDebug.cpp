/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/VisualDebug.hpp>
#include <cctag/utils/FileDebug.hpp>
#include <cctag/Plane.hpp>

#include <boost/filesystem.hpp>
#include "cctag/Plane.hpp"

namespace bfs = boost::filesystem;

namespace cctag
{

CCTagVisualDebug::CCTagVisualDebug() = default;

CCTagVisualDebug::~CCTagVisualDebug() = default;

void CCTagVisualDebug::initializeFolders(const boost::filesystem::path & rootPath, const std::string & outputFolder, std::size_t nCrowns)
{
#ifdef CCTAG_SERIALIZE
  // Create inputImagePath/result if it does not exist
  std::stringstream resultFolderName, localizationFolderName, identificationFolderName, absoluteOutputFolderName;
  
  resultFolderName << rootPath.string();
  
  if( outputFolder != "" ) {
    resultFolderName << "/" << outputFolder;
    if (!bfs::exists(resultFolderName.str())) {
      bfs::create_directory(resultFolderName.str());
    }
  }
  
  resultFolderName << "/cctag" << nCrowns << "CC";
  
  if (!bfs::exists(resultFolderName.str())) {
      bfs::create_directory(resultFolderName.str());
  }
  
  localizationFolderName << resultFolderName.str() << "/localization";
  identificationFolderName << resultFolderName.str() << "/identification";

  if (!bfs::exists(localizationFolderName.str())) {
      bfs::create_directory(localizationFolderName.str());
  }

  if (!bfs::exists(identificationFolderName.str())) {
      bfs::create_directory(identificationFolderName.str());
  }
  
  _pathRoot = resultFolderName.str();
  CCTAG_COUT_VAR(_pathRoot);
#endif
}

void CCTagVisualDebug::setPyramidLevel(int level) {
#ifdef CCTAG_SERIALIZE
  _pyramidLevel = level;
#endif
}

int CCTagVisualDebug::getPyramidLevel() const {
    return _pyramidLevel;
}

void CCTagVisualDebug::resetMarkerIndex() 
{
#ifdef CCTAG_SERIALIZE
  _markerIndex = 0;;
#endif // CCTAG_SERIALIZE
}

#ifdef CCTAG_SERIALIZE
void CCTagVisualDebug::incrementMarkerIndex() 
{
  ++_markerIndex;
}

std::size_t CCTagVisualDebug::getMarkerIndex() const
{
  return _markerIndex;
}
#endif // CCTAG_SERIALIZE

std::string CCTagVisualDebug::getPath() const {
  return _path;
}

void CCTagVisualDebug::setImageFileName(const std::string& imageFileName) {
#ifdef CCTAG_SERIALIZE
  _imageFileName = imageFileName;
    CCTAG_COUT_VAR(_imageFileName);
  _path = _pathRoot + "/" + imageFileName;
  CCTAG_COUT_VAR(_path);
  if (!bfs::exists(_path)) {
    bfs::create_directory(_path);
    CCTAG_COUT("creation done");
  }
  CCTAG_COUT("exit");
#endif   
}

void CCTagVisualDebug::initBackgroundImage(const Plane<uint8_t>& b)
{
#ifdef CCTAG_SERIALIZE
    int h = b.getRows();
    int w = b.getCols();
    _backImage = Plane<Color>( h, w );
    for( int y=0; y<h; y++ )
    {
        for( int x=0; x<w; x++ )
        {
            float val = b.at(x,y) / 255.0f;
            _backImage.at(x,y) = Color( val, val, val, 0 );
        }
    }
#endif
}

void CCTagVisualDebug::newSession(const std::string & sessionName) {
#ifdef CCTAG_SERIALIZE
  // Don't erase old sessions
  if (_sessions.find(sessionName) == _sessions.end())
  {
      _sessions[sessionName] = _backImage;
  }
  else
  {
    _backImage = _sessions[sessionName];
  }
#endif
}

#if 0
void CCTagVisualDebug::drawText(const cctag::Point2d<Eigen::Vector3f> & p, const std::string & text, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
  CvFont font1;
  cvInitFont(&font1, CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.8, 0, 2);

  IplImage iplBack = planeToMat(_backImage);
  cvPutText( &iplBack, text.c_str(),
          cvPoint((int) p.x(), (int) p.y()),
          &font1, CV_RGB(color[0] * 255, color[1] * 255, color[2] * 255));
#endif
}
#endif

void CCTagVisualDebug::drawPoint(const float x, const float y, const cctag::Color & color )
{
#ifdef CCTAG_SERIALIZE
    if( x >= 1.0f && x < _backImage.getCols()-1.0f &&
        y >= 1.0f && y < _backImage.getRows()-1.0f )
    {
        _backImage.at( (int)roundf(x), (int)roundf(y) ) = color;
    }
#endif // CCTAG_SERIALIZE
}

void CCTagVisualDebug::drawPoint(const cctag::Point2d<Eigen::Vector3f> & point, const cctag::Color & color)
{
#ifdef CCTAG_SERIALIZE
    drawPoint( point.x(), point.y(), color );
#endif // CCTAG_SERIALIZE
}

void CCTagVisualDebug::drawPoint(const cctag::DirectedPoint2d<Eigen::Vector3f> & point, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
  if (point.x() >= 1 && point.x() < _backImage.getCols()-1 &&
          point.y() >= 1 && point.y() < _backImage.getRows()-1)
  {
    const float xlenf = point.dX();
    const float ylenf = point.dY();
    if( fabsf(xlenf) >= fabsf(ylenf) )
    {
        int   xlen  = xlenf;
        float ystep = ylenf / xlenf;
        for( int x=0; x<xlen; x++ )
        {
            _backImage.at(point.x()+x,point.y()+x*ystep) = color;
        }
    }
    else
    {
        int   ylen  = ylenf;
        float xstep = xlenf / ylenf;
        for( int y=0; y<ylen; y++ )
        {
            _backImage.at(point.x()+y*xstep,point.y()+y) = color;
        }
    }
  }
#endif // CCTAG_SERIALIZE
}

void CCTagVisualDebug::drawPoints(const std::vector<cctag::Point2d<Eigen::Vector3f> > & points, const cctag::Color & color)
{
#ifdef CCTAG_SERIALIZE
  for(const cctag::Point2d<Eigen::Vector3f> & point : points) {
      CCTagVisualDebug::instance().drawPoint(point, cctag::color_red);
  }
#endif
}

// todo@Lilian: template that function
void CCTagVisualDebug::drawPoints(const std::vector<cctag::DirectedPoint2d<Eigen::Vector3f> > & points, const cctag::Color & color)
{
#ifdef CCTAG_SERIALIZE
  for(const cctag::Point2d<Eigen::Vector3f> & point : points) {
      CCTagVisualDebug::instance().drawPoint(cctag::Point2d<Eigen::Vector3f>(point.x(),point.y()), cctag::color_red);
  }
#endif
}

void CCTagVisualDebug::drawMarker(const cctag::CCTag& marker, bool drawScaledMarker)
{
#ifdef CCTAG_SERIALIZE
  numerical::geometry::Ellipse rescaledOuterEllipse;
  if (drawScaledMarker) {
      rescaledOuterEllipse = marker.rescaledOuterEllipse();
  } else {
      rescaledOuterEllipse = marker.outerEllipse();
  }
  Point2d<Eigen::Vector3f> & center = rescaledOuterEllipse.center();
  
  // Display ellipses
  if (drawScaledMarker) {
      rescaledOuterEllipse = marker.rescaledOuterEllipse();
  } else {
      rescaledOuterEllipse = marker.outerEllipse();
  }

  Color color;
  // Set the color
  if (marker.getStatus() == status::no_collected_cuts) {
    // Magenta
    color = Color(1,0,1);
  }else if (marker.getStatus() == status::no_selected_cuts) {
    // Cyan
    color = Color(0,1,1);
  }else if(marker.getStatus() == status::opti_has_diverged){
    // Red
    color = Color(1,0,0);
  }else if(marker.getStatus() == status::id_not_reliable){
    // Cyan
    color = Color(0,1,1);
  }else if(marker.getStatus() == status::id_reliable){
    // Green
    color = Color(0,1,0);
  }else if(marker.getStatus() == status::degenerate){
    // Yellow 1
    color = Color(1,1,0);
  }else if(marker.getStatus() == 0 ){
    // Green
    color = Color(0,1,0);
  }

    float xc = center.x();
    float yc = center.y();
    float a  = rescaledOuterEllipse.a();
    float b  = rescaledOuterEllipse.b();
    float t  = boost::math::constants::pi<double>() / 180.0;
    float alpha = rescaledOuterEllipse.angle();
  
    // Filling each pixel corresponding 
    // to every angle from 0 to 360 
    for (int theta = 0; theta < 360; theta += 1)
    { 
        int x = a * cos(t * theta) * cos(alpha) 
                + b * sin(t * theta) * sin(alpha); 
  
        int y = b * sin(t * theta) * cos(alpha) 
                - a * cos(t * theta) * sin(alpha); 
  
        drawPoint(xc + x, yc - y, color); 
    } 
#endif
}

void CCTagVisualDebug::drawInfos(const cctag::CCTag& marker, bool drawScaledMarker)
{
#if 0
#ifdef CCTAG_SERIALIZE
    CvFont font1;
  cvInitFont(&font1, CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.8, 0, 2);

  std::string sId = boost::lexical_cast<std::string>(marker.id() + 1);

  int x, y;
  if (drawScaledMarker) {
      x = int (marker.rescaledOuterEllipse().center().x());
      y = int (marker.rescaledOuterEllipse().center().y());
  } else {
      x = int (marker.outerEllipse().center().x());
      y = int (marker.outerEllipse().center().y());
  }

  IplImage iplImg = planeToMat( _backImage );
  cvPutText( &iplImg, sId.c_str(),
          cvPoint(x-10, y+10),
          &font1, CV_RGB(255, 140, 0));
#endif
#endif
}

std::string CCTagVisualDebug::getImageFileName() const {
    return _imageFileName;
}

void CCTagVisualDebug::out(const std::string & filename) const
{
#if defined(CCTAG_SERIALIZE) && defined(CCTAG_VISUAL_DEBUG)
    writePlanePPM( filename, _backImage, SCALED_WRITING );
#endif
}

void CCTagVisualDebug::outPutAllSessions() const
{
#if defined(CCTAG_SERIALIZE) && defined(CCTAG_VISUAL_DEBUG)
    std::cerr << "Called outPutAllSessions" << std::endl;

    for(const Sessions::const_iterator::value_type & v : _sessions) {
        const std::string filename = _path + "/" + v.first + ".ppm";
        writePlanePPM( filename, v.second, SCALED_WRITING );
    }
#endif
}

void CCTagVisualDebug::writeLocalizationView(cctag::CCTag::List& markers) const {
#ifdef CCTAG_SERIALIZE
    std::stringstream localizationResultFileName;
    localizationResultFileName << "../localization/" << _imageFileName;
    CCTagVisualDebug::instance().newSession(localizationResultFileName.str());

    for(const cctag::CCTag & marker : markers) {
        CCTagVisualDebug::instance().drawMarker(marker);
        CCTagVisualDebug::instance().drawInfos(marker);
    }
#endif
}

void CCTagVisualDebug::writeIdentificationView(cctag::CCTag::List& markers) const {
#ifdef CCTAG_SERIALIZE

    std::stringstream identificationResultFileName;
    identificationResultFileName << "../identification/" << _imageFileName;
    CCTagVisualDebug::instance().newSession(identificationResultFileName.str());

    for(const cctag::CCTag & marker : markers) {
        CCTagVisualDebug::instance().drawMarker(marker);
        CCTagVisualDebug::instance().drawPoints(marker.rescaledOuterEllipsePoints(), cctag::color_red);
        CCTagVisualDebug::instance().drawInfos(marker);
    }

#endif
}

void CCTagVisualDebug::clearSessions() {
#ifdef CCTAG_SERIALIZE
    _sessions.erase(_sessions.begin(), _sessions.end());
#endif
}

} // namespace cctag
