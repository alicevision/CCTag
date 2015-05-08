#ifndef _CCTAG_CCTAG_VISUALDEBUG_HPP_
#define	_CCTAG_CCTAG_VISUALDEBUG_HPP_

#include <cctag/draw.hpp>
#include <cctag/progBase/pattern/Singleton.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/colors.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/boostCv/cvImage.hpp>

#include <boost/gil/image_view.hpp>


namespace cctag
{

class CCTagVisualDebug : public Singleton<CCTagVisualDebug> {
    MAKE_SINGLETON_WITHCONSTRUCTORS(CCTagVisualDebug)

public:
    typedef std::map<std::string, boost::gil::rgb8_image_t> Sessions;
public:

    void setPyramidLevel(int level);

    int getPyramidLevel();

    void initPath(const std::string & path);

    void setImageFileName(const std::string& imageFileName);

    template<class SView>
    void initBackgroundImage(const SView & backView) {
#ifdef CCTAG_SERIALIZE
        using namespace boost::gil;
        _backImage.recreate(backView.width(), backView.height());
        _backView = boost::gil::view(_backImage);
        copy_and_convert_pixels(backView, _backView);
#endif
    }

    void newSession(const std::string & sessionName);

    void changeSession(const std::string & sessionName);

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

    boost::gil::rgb8_view_t _view; ///< Current view
    boost::gil::rgb8_image_t _backImage; ///< Background image
    boost::gil::rgb8_view_t _backView; ///< Background view

    int _pyramidLevel;

    std::string _path;
    std::string _imageFileName;
};

} // namespace cctag


#endif

