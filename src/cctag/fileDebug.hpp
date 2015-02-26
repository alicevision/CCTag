#ifndef _CCTAG_CCTAGOUTPUT_HPP_
#define	_CCTAG_CCTAGOUTPUT_HPP_

#include "modeConfig.hpp"

#include <cctag/global.hpp>
#include <cctag/progBase/pattern/Singleton.hpp>
#include <cctag/CCTag.hpp>

#include <boost/ptr_container/ptr_map.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <fstream>
#include <string>
#include <sstream>

#define NOT_IN_RESEARCH_AREA 1
#define FLOW_LENGTH 2
#define SAME_LABEL 3
#define PTS_OUT_WHILE_ASSEMBLING 4
#define BAD_GRAD_WHILE_ASSEMBLING 5
#define FINAL_MEDIAN_TEST_FAILED_WHILE_ASSEMBLING 6
#define QUALITY_TEST_FAILED_WHILE_ASSEMBLING 7
#define MEDIAN_TEST_FAILED_WHILE_ASSEMBLING 8
#define PTSOUTSIDE_OR_BADGRADORIENT 9
#define RATIO_SEMIAXIS 10
#define PTS_OUTSIDE_ELLHULL 11
#define RAISED_EXCEPTION 12
#define PASS_ALLTESTS 30

namespace rom {
    namespace vision {

        class CCTagFileDebug : public Singleton<CCTagFileDebug> {
            MAKE_SINGLETON_WITHCONSTRUCTORS(CCTagFileDebug)

        public:
            typedef boost::ptr_map<std::string, std::stringstream> Sessions;
        public:

            void setPath(const std::string& folderName);
            
            void newSession(const std::string& sessionName);

            void outputFlowComponentAssemblingInfos(int status);
            
            void initFlowComponentsIndex(int size);
            void resetFlowComponent();
            void incrementFlowComponentIndex(int n);
            void setResearchArea(rom::numerical::geometry::Ellipse circularResearchArea);
            void setFlowComponentAssemblingState( bool isAssembled, int indexSelectedFlowComponent);
            void outputFlowComponentInfos(const rom::vision::marker::CCTagFlowComponent & flowComponent);
            
            void outputMarkerInfos(const rom::vision::marker::CCTag& marker);

            void outPutAllSessions() const;

            void clearSessions();
            
            void printInfos();

            
            
        private:
            Sessions _sessions; ///< Sessions map

            std::stringstream *_sstream; ///< Current string stream
            std::string _path;
            
            // flowComponents index used in detection.
            std::vector<int> _vflowComponentIndex;
            bool _isAssembled;
            rom::numerical::geometry::Ellipse _researchArea;
        };
    }
}


#endif

