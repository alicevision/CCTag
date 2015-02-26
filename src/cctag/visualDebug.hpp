#ifndef _CCTAG_CCTAG_VISUALDEBUG_HPP_
#define	_CCTAG_CCTAG_VISUALDEBUG_HPP_

#include "modeConfig.hpp"

#include <boost/gil/image_view.hpp>

#include "draw.hpp"

#include <cctag/progBase/pattern/Singleton.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/colors.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/boostCv/cvImage.hpp>


namespace popart {
    namespace vision {

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
#if defined(DEBUG) || defined(CCTAG_STAT_DEBUG)
                using namespace boost::gil;
                _backImage.recreate(backView.width(), backView.height());
                _backView = boost::gil::view(_backImage);
                copy_and_convert_pixels(backView, _backView);
#endif
            }

            void newSession(const std::string & sessionName);

            void changeSession(const std::string & sessionName);

            void drawText(const popart::Point2dN<double> & p, const std::string & text, const popart::Color & color);

            void drawPoint(const popart::Point2dN<double> & p, const popart::Color & color);

            void drawPoints(const std::vector<popart::Point2dN<double> > & pts, const popart::Color & color);

            void drawMarker(const popart::vision::marker::CCTag& marker, bool drawScaledMarker = true);

            void drawInfos(const popart::vision::marker::CCTag& marker, bool drawScaledMarker = true);

            void out(const std::string & filename) const;

            void outPutAllSessions() const;

            void writeLocalizationView(popart::vision::marker::CCTag::List & markers) const;

            void writeIdentificationView(popart::vision::marker::CCTag::List & markers) const;

            std::string getImageFileName() const;

        private:
            Sessions _sessions; ///< Sessions map

            boost::gil::rgb8_view_t _view; ///< Current view
            boost::gil::rgb8_image_t _backImage; ///< Background image
            boost::gil::rgb8_view_t _backView; ///< Background view

            int _pyramidLevel;

            std::string _path;
            std::string _imageFileName;
        };

    }
}


#endif

