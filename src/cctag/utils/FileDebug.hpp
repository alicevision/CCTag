/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_CCTAGOUTPUT_HPP_
#define	_CCTAG_CCTAGOUTPUT_HPP_

#include <cctag/utils/Defines.hpp>
#include <cctag/utils/Singleton.hpp>
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

namespace cctag {

        class CCTagFileDebug : public Singleton<CCTagFileDebug> {
            MAKE_SINGLETON_WITHCONSTRUCTORS(CCTagFileDebug)

        public:
            using Sessions = boost::ptr_map<std::string, std::stringstream>;
        public:

            void setPath(const std::string& folderName);
            
            void newSession(const std::string& sessionName);

            void outputFlowComponentAssemblingInfos(int status);
            
            void initFlowComponentsIndex(int size);
            void resetFlowComponent();
            void incrementFlowComponentIndex(int n);
            void setResearchArea(const cctag::numerical::geometry::Ellipse& circularResearchArea);
            void setFlowComponentAssemblingState( bool isAssembled, int indexSelectedFlowComponent);
            void outputFlowComponentInfos(const cctag::CCTagFlowComponent & flowComponent);
            
            void outputMarkerInfos(const cctag::CCTag& marker);

            void outPutAllSessions() const;

            void clearSessions();
            
            void printInfos();
            
            // Vote debug
            void newVote(float x, float y, float dx, float dy);
            void addFieldLinePoint(float x, float y);
            void endVote();

            
            
        private:
            Sessions _sessions; ///< Sessions map

            std::stringstream *_sstream; ///< Current string stream
            std::string _path;
            
            // flowComponents index used in detection.
            std::vector<int> _vflowComponentIndex;
            bool _isAssembled;
            cctag::numerical::geometry::Ellipse _researchArea;
        };
        
} // namespace cctag


#endif

