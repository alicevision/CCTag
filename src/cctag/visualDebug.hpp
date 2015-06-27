#ifndef _CCTAG_CCTAG_VISUALDEBUG_HPP_
#define	_CCTAG_CCTAG_VISUALDEBUG_HPP_

#include <cctag/draw.hpp>
#include <cctag/progBase/pattern/Singleton.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/colors.hpp>
#include <cctag/CCTag.hpp>

namespace cctag
{

class CCTagVisualDebug : public Singleton<CCTagVisualDebug> {
    MAKE_SINGLETON_WITHCONSTRUCTORS(CCTagVisualDebug)

public:
    typedef std::map<std::string, cv::Mat> Sessions;
public:

    void setPyramidLevel(int level);

    int getPyramidLevel();
    
    std::string getPath() const;

    void setImageFileName(const std::string& imageFileName);

    void initBackgroundImage(const cv::Mat & back);
    
    void initializeFolders(const std::string & filename, std::size_t nCrowns = 4);

    void newSession(const std::string & sessionName);

    void drawText(const cctag::Point2dN<double> & p, const std::string & text, const cctag::Color & color);

    void drawPoint(const cctag::Point2dN<double> & p, const cctag::Color & color);

    void drawPoints(const std::vector<cctag::Point2dN<double> > & pts, const cctag::Color & color);

    void drawMarker(const cctag::CCTag& marker, bool drawScaledMarker = true);

    void drawInfos(const cctag::CCTag& marker, bool drawScaledMarker = true);

    void out(const std::string & filename) const;

    void outPutAllSessions() const;

    void writeLocalizationView(cctag::CCTag::List & markers) const;

    void writeIdentificationView(cctag::CCTag::List & markers) const;

    std::string getImageFileName() const;

    void clearSessions();

private:
    Sessions _sessions; ///< Sessions map

    cv::Mat _backImage; // Background image
    int _pyramidLevel;
    std::string _imageFileName;
    std::string _pathRoot;
    std::string _path;
};

} // namespace cctag


#endif

