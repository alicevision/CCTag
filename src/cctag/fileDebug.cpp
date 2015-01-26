#include "fileDebug.hpp"
#include "CCTagFlowComponent.hpp"
#include "dataSerialization.hpp"

namespace rom {
    namespace vision {

        CCTagFileDebug::CCTagFileDebug()
        : _sstream(NULL) {

        }

        CCTagFileDebug::~CCTagFileDebug() {

        }

        void CCTagFileDebug::setPath(const std::string& folderName) {
#ifdef CCTAG_STAT_DEBUG
            // Don't erase old sessions
            _path = folderName;
#endif
        }

        void CCTagFileDebug::newSession(const std::string& sessionName) {
#ifdef CCTAG_STAT_DEBUG
            // Don't erase old sessions
            Sessions::iterator it = _sessions.find(sessionName);
            if (it == _sessions.end()) {
                _sstream = &_sessions[sessionName];
            } else {
                _sstream = it->second;
            }
#endif
        }

        void CCTagFileDebug::outputFlowComponentAssemblingInfos(int status) {
#if defined CCTAG_STAT_DEBUG && defined DEBUG
            boost::archive::text_oarchive oa(*_sstream);

            BOOST_FOREACH(const int index, _vflowComponentIndex) {
                oa & BOOST_SERIALIZATION_NVP(index);
            }
            oa & BOOST_SERIALIZATION_NVP(status);
            const bool aux = _isAssembled;
            oa & BOOST_SERIALIZATION_NVP(aux);
            rom::vision::marker::serializeEllipse(oa, _researchArea);
            _researchArea = rom::numerical::geometry::Ellipse();
#endif
        }

        void CCTagFileDebug::printInfos() {
            ROM_COUT("Print infos");
            BOOST_FOREACH(const int index, _vflowComponentIndex) {
                ROM_COUT_VAR(index);
            }
            ROM_COUT_VAR(_isAssembled);
            ROM_PAUSE;
        }

        void CCTagFileDebug::initFlowComponentsIndex(int size) {
#if defined CCTAG_STAT_DEBUG && defined DEBUG
            _vflowComponentIndex.resize(size);
            for (int i = 0; i < _vflowComponentIndex.size(); ++i) {
                _vflowComponentIndex[i] = 0;
            }
#endif            
        }

        void CCTagFileDebug::resetFlowComponent() {
#if defined CCTAG_STAT_DEBUG && defined DEBUG
            _isAssembled = false;
            _researchArea = rom::numerical::geometry::Ellipse();

            if (_vflowComponentIndex.size() > 1) {
                for (int i = 1; i < _vflowComponentIndex.size(); ++i) {
                    _vflowComponentIndex[i] = 0;
                }
            }
#endif            
        }

        void CCTagFileDebug::incrementFlowComponentIndex(int n) {
#if defined CCTAG_STAT_DEBUG && defined DEBUG
            (_vflowComponentIndex[n])++;
#endif            
        }

        void CCTagFileDebug::setResearchArea(rom::numerical::geometry::Ellipse circularResearchArea) {
#if defined CCTAG_STAT_DEBUG && defined DEBUG
            _researchArea = circularResearchArea;
#endif 
        }

        void CCTagFileDebug::setFlowComponentAssemblingState(bool isAssembled, int indexSelectedFlowComponent) {
#if defined CCTAG_STAT_DEBUG && defined DEBUG
            _isAssembled = isAssembled;
            _vflowComponentIndex[1] = indexSelectedFlowComponent;
#endif
        }

        void CCTagFileDebug::outputFlowComponentInfos(const rom::vision::marker::CCTagFlowComponent & flowComponent) {
#ifdef CCTAG_STAT_DEBUG
            if (_sstream) {
                boost::archive::text_oarchive oa(*_sstream);
                //oa << flowComponent;
                const std::size_t nCircles = flowComponent._nCircles;
                oa & BOOST_SERIALIZATION_NVP(nCircles);
                rom::vision::marker::serializeFlowComponent(oa, flowComponent);
            } else {
                ROM_COUT_ERROR("Unable to output flowComponent infos! Select session before!");
            }
#endif
        }

        void CCTagFileDebug::outputMarkerInfos(const rom::vision::marker::CCTag& marker) {
#ifdef CCTAG_STAT_DEBUG
            if (_sstream) {
                boost::archive::text_oarchive oa(*_sstream);
                oa << marker;
            } else {
                ROM_COUT_ERROR("Unable to output marker infos! Select session before!");
            }
#endif
        }

        void CCTagFileDebug::outPutAllSessions() const {
#ifdef CCTAG_STAT_DEBUG
            for (Sessions::const_iterator it = _sessions.begin(), itEnd = _sessions.end(); it != itEnd; ++it) {
                const std::string filename = _path + "/" + it->first; //cctagFileDebug_
                //const std::string filename = it->first + "/data_v2.txt";
                std::ofstream f(filename.c_str());
                f << it->second->str();
            }
#endif
        }

        void CCTagFileDebug::clearSessions() {
#ifdef CCTAG_STAT_DEBUG
            _sessions.erase(_sessions.begin(), _sessions.end());
#endif
        }

    }
}

