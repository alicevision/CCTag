#ifndef _CCTAG_CCTAG_VISUALDEBUG_HPP_
#define	_CCTAG_CCTAG_VISUALDEBUG_HPP_

#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/progBase/Singleton.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/Colors.hpp>
#include <cctag/CCTag.hpp>

#include <boost/filesystem.hpp>

namespace cctag
{

class CCTagVisualDebug : public Singleton<CCTagVisualDebug> {
    MAKE_SINGLETON_WITHCONSTRUCTORS(CCTagVisualDebug)

public:
    typedef std::map<std::string, cv::Mat> Sessions;
public:

    void setPyramidLevel(int level);

    int getPyramidLevel();
    
    void resetMarkerIndex();

    void incrementMarkerIndex();
    
    std::size_t getMarkerIndex();
    
    std::string getPath() const;

    void setImageFileName(const std::string& imageFileName);

    void initBackgroundImage(const cv::Mat & back);
    
    void initializeFolders(const boost::filesystem::path & filename, const std::string & outputFolder, std::size_t nCrowns = 4);

    void newSession(const std::string & sessionName);

    void drawText(const cctag::Point2dN<double> & p, const std::string & text, const cctag::Color & color);

    void drawPoint(const cctag::Point2dN<double> & p, const cctag::Color & color);
    void drawPoint(const cctag::DirectedPoint2d<double> & point, const cctag::Color & color);

    void drawPoints(const std::vector<cctag::Point2dN<double> > & pts, const cctag::Color & color);
    // todo templater la function ci-dessus avec celle ci-dessous
    void drawPoints(const std::vector<cctag::DirectedPoint2d<double> > & points, const cctag::Color & color);

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

