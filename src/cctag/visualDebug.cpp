#include <cctag/visualDebug.hpp>
#include <cctag/fileDebug.hpp>

#include <boost/filesystem.hpp>

// griff: define int_p_NULL against a bug in boost-gil-numeric 1.0.0
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

namespace bfs = boost::filesystem;

namespace cctag
{

CCTagVisualDebug::CCTagVisualDebug() {

}

CCTagVisualDebug::~CCTagVisualDebug() {

}

void CCTagVisualDebug::initializeFolders(const boost::filesystem::path & rootPath, std::size_t nCrowns)
{
#ifdef CCTAG_SERIALIZE
  // Create inputImagePath/result if it does not exist
  std::stringstream resultFolderName, localizationFolderName, identificationFolderName;
  resultFolderName << rootPath.string() << "/cctag" << nCrowns << "CC";
  localizationFolderName << resultFolderName.str() << "/localization";
  identificationFolderName << resultFolderName.str() << "/identification";

  if (!bfs::exists(resultFolderName.str())) {
      bfs::create_directory(resultFolderName.str());
  }

  if (!bfs::exists(localizationFolderName.str())) {
      bfs::create_directory(localizationFolderName.str());
  }

  if (!bfs::exists(identificationFolderName.str())) {
      bfs::create_directory(identificationFolderName.str());
  }
  
  _pathRoot = resultFolderName.str();
  CCTAG_COUT_VAR(_pathRoot);
#endif
    // std::cerr << "Leave " << __FUNCTION__ << std::endl;
}

void CCTagVisualDebug::setPyramidLevel(int level) {
#ifdef CCTAG_SERIALIZE
    _pyramidLevel = level;
#endif
}

int CCTagVisualDebug::getPyramidLevel() {
    return _pyramidLevel;
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

void CCTagVisualDebug::newSession(const std::string & sessionName) {
#ifdef CCTAG_SERIALIZE
    using namespace boost::gil;
    // Don't erase old sessions
    if (_sessions.find(sessionName) == _sessions.end()) {
        _sessions[sessionName].recreate(_backView.width(), _backView.height());
        copy_pixels(_backView, view(_sessions[sessionName]));
    }
    _view = view(_sessions[sessionName]);
#endif
}

void CCTagVisualDebug::changeSession(const std::string & sessionName) {
#ifdef CCTAG_SERIALIZE 
    using namespace boost::gil;
    _view = view(_sessions[sessionName]);
#endif
}

void CCTagVisualDebug::drawText(const cctag::Point2dN<double> & p, const std::string & text, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
    using namespace boost::gil;
    boostCv::CvImageView cvview(_view);
    IplImage * img = cvview.get();
    CvFont font1;
    cvInitFont(&font1, CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.8, 0, 2);

    cvPutText(img, text.c_str(),
            cvPoint((int) p.x(), (int) p.y()),
            &font1, CV_RGB(color[0] * 255, color[1] * 255, color[2] * 255));
#endif
}

void CCTagVisualDebug::drawPoint(const cctag::Point2dN<double> & p, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE
    using namespace boost::gil;
    if (p.x() >= 0.0 && p.x() < _view.width() &&
            p.y() >= 0.0 && p.y() < _view.height()) {
        rgb8_view_t::xy_locator loc = _view.xy_at(p.x(), p.y());
        rgb32f_pixel_t rgb;
        get_color(rgb, red_t()) = color[0];
        get_color(rgb, green_t()) = color[1];
        get_color(rgb, blue_t()) = color[2];
        color_convert(rgb, *loc);
    } else {
        //CCTAG_TCOUT_VAR2( p.x(), _view.width() ); // todo@Eloi, p.x(): 23, _view.width(): 0, p.y(): 38, _view.height(): 0 ?? apparait tres souvent
        //CCTAG_TCOUT_VAR2( p.y(), _view.height() );
    }
#endif
}

void CCTagVisualDebug::drawPoints(const std::vector<cctag::Point2dN<double> > & pts, const cctag::Color & color) {
#ifdef CCTAG_SERIALIZE

    BOOST_FOREACH(const cctag::Point2dN<double> & p, pts) {
        CCTagVisualDebug::instance().drawPoint(p, cctag::color_red);
    }
#endif
}

void CCTagVisualDebug::drawMarker(const cctag::CCTag& marker, bool drawScaledMarker) {
#ifdef CCTAG_SERIALIZE
    cctag::drawMarkerOnGilImage(_view, marker, drawScaledMarker);
#endif
}

void CCTagVisualDebug::drawInfos(const cctag::CCTag& marker, bool drawScaledMarker) {
#ifdef CCTAG_SERIALIZE
    cctag::drawMarkerInfos(_view, marker, drawScaledMarker);
#endif
}

std::string CCTagVisualDebug::getImageFileName() const {
    return _imageFileName;
}

void CCTagVisualDebug::out(const std::string & filename) const {
#ifdef CCTAG_SERIALIZE
    boost::gil::png_write_view(filename, _view);
#endif
}

void CCTagVisualDebug::outPutAllSessions() const {
#ifdef CCTAG_SERIALIZE
    using namespace boost::gil;

    BOOST_FOREACH(const Sessions::const_iterator::value_type & v, _sessions) {
        const std::string filename = _path + "/" + v.first + ".png";
        const rgb8c_view_t svw = const_view(v.second);
        //boost::gil::jpeg_write_view( filename, svw );
        boost::gil::png_write_view(filename, svw);
        //CCTAG_COUT_VAR(filename);

    }
#endif
}

void CCTagVisualDebug::writeLocalizationView(cctag::CCTag::List& markers) const {
#ifdef CCTAG_SERIALIZE

    std::stringstream localizationResultFileName;
    localizationResultFileName << "../localization/" << _imageFileName;
    CCTagVisualDebug::instance().newSession(localizationResultFileName.str());

    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
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

    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
        CCTagVisualDebug::instance().drawMarker(marker);
        CCTagVisualDebug::instance().drawPoints(marker.rescaledOuterEllipsePoints(), cctag::color_red);
        //CCTagVisualDebug::instance().draw
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
