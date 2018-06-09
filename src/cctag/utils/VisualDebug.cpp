/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/VisualDebug.hpp>
#include <cctag/utils/FileDebug.hpp>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

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
#endif
}

void CCTagVisualDebug::incrementMarkerIndex() 
{
#ifdef CCTAG_SERIALIZE
  ++_markerIndex;
#endif
}

std::size_t CCTagVisualDebug::getMarkerIndex() const
{
#ifdef CCTAG_SERIALIZE
  return _markerIndex;
#else
	return 0;
#endif
}

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

void CCTagVisualDebug::initBackgroundImage(const cv::Mat & back)
{
#ifdef CCTAG_SERIALIZE
  cv::Mat temp;
  cvtColor(back, temp, cv::COLOR_GRAY2RGB);
  _backImage = temp.clone();
#endif
}

void CCTagVisualDebug::newSession(const std::string & sessionName) {
#ifdef CCTAG_SERIALIZE
  // Don't erase old sessions
  if (_sessions.find(sessionName) == _sessions.end()) {
      _sessions[sessionName] = _backImage;
  }else
  {
    _backImage = _sessions[sessionName];
  }
#endif
}

void CCTagVisualDebug::drawText(const cctag::Point2d<Eigen::Vector3f> & p, const std::string & text, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
  CvFont font1;
  cvInitFont(&font1, CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.8, 0, 2);

  IplImage iplBack = _backImage;
  cvPutText( &iplBack, text.c_str(),
          cvPoint((int) p.x(), (int) p.y()),
          &font1, CV_RGB(color[0] * 255, color[1] * 255, color[2] * 255));
#endif
}

void CCTagVisualDebug::drawPoint(const cctag::Point2d<Eigen::Vector3f> & point, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
  if (point.x() >= 1 && point.x() < _backImage.cols-1 &&
          point.y() >= 1 && point.y() < _backImage.rows-1)
  {
    cv::Vec3b cvColor;
    cvColor.val[0] = 255*color[0];
    cvColor.val[1] = 255*color[1]; 
    cvColor.val[2] = 255*color[2]; 
    _backImage.at<cv::Vec3b>(point.y(),point.x()) = cvColor;
    //cv::rectangle(_backImage, cvPoint(point.x()-1.f,point.y()-1.f), cvPoint(point.x()+1.f,point.y()+1.f), cv::Scalar(255*color[0], 255*color[1], 255*color[2]),0);
  }
#endif // CCTAG_SERIALIZE
}

void CCTagVisualDebug::drawPoint(const cctag::DirectedPoint2d<Eigen::Vector3f> & point, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
  if (point.x() >= 1 && point.x() < _backImage.cols-1 &&
          point.y() >= 1 && point.y() < _backImage.rows-1)
  {
    //cv::Vec3b cvColor;
    //cvColor.val[0] = 255*color[0];
    //cvColor.val[1] = 255*color[1]; 
    //cvColor.val[2] = 255*color[2]; 
    //_backImage.at<cv::Vec3b>(point.y(),point.x()) = cvColor;
    cv::Point p1(point.x(),point.y());
    cv::Point p2(point.x() + point.dX(),point.y() + point.dY());
    cv::arrowedLine( _backImage, p1, p2, cv::Scalar(255*color[0], 255*color[1], 255*color[2]) );
    
    //cv::rectangle(_backImage, cvPoint(point.x()-1.f,point.y()-1.f), cvPoint(point.x()+1.f,point.y()+1.f), cv::Scalar(255*color[0], 255*color[1], 255*color[2]),0);
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

  cv::Scalar color;
  // Set the color
  if (marker.getStatus() == status::no_collected_cuts) {
    // Magenta
    color = cv::Scalar(255,0,255);
  }else if (marker.getStatus() == status::no_selected_cuts) {
    // Cyan
    color = cv::Scalar(0,255,255);
  }else if(marker.getStatus() == status::opti_has_diverged){
    // Red
    color = cv::Scalar(255,0,0);
  }else if(marker.getStatus() == status::id_not_reliable){
    // Cyan
    color = cv::Scalar(0,255,255);
  }else if(marker.getStatus() == status::id_reliable){
    // Green
    color = cv::Scalar(0,255,0);
  }else if(marker.getStatus() == status::degenerate){
    // Yellow 1
    color = cv::Scalar(255,255,0);
  }else if(marker.getStatus() == 0 ){
    // Green
    color = cv::Scalar(0,255,0);
  }

  //CCTAT_COUT_VAR(color);
  
  cv::ellipse(_backImage , cv::Point(center.x(),center.y()),
      cv::Size(rescaledOuterEllipse.a(), rescaledOuterEllipse.b()),
      rescaledOuterEllipse.angle()*180/boost::math::constants::pi<double>(), 0, 360, color);
#endif
}

void CCTagVisualDebug::drawInfos(const cctag::CCTag& marker, bool drawScaledMarker)
{
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

  IplImage iplImg = _backImage;
  cvPutText( &iplImg, sId.c_str(),
          cvPoint(x-10, y+10),
          &font1, CV_RGB(255, 140, 0));
#endif
}

std::string CCTagVisualDebug::getImageFileName() const {
    return _imageFileName;
}

void CCTagVisualDebug::out(const std::string & filename) const {
#if defined(CCTAG_SERIALIZE) && defined(VISUAL_DEBUG)
  cv::imwrite(filename, _backImage);
#endif
}

void CCTagVisualDebug::outPutAllSessions() const {
#if defined(CCTAG_SERIALIZE) && defined(VISUAL_DEBUG)
    for(const Sessions::const_iterator::value_type & v : _sessions) {
        const std::string filename = _path + "/" + v.first + ".png";
        cv::imwrite(filename, v.second);
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
