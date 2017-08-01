/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/FileDebug.hpp>
#include <cctag/CCTagFlowComponent.hpp>
#include <cctag/DataSerialization.hpp>

#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

namespace cctag
{

CCTagFileDebug::CCTagFileDebug()
: _sstream(nullptr) {

}

CCTagFileDebug::~CCTagFileDebug() = default;

void CCTagFileDebug::setPath(const std::string& folderName)
{
#ifdef CCTAG_SERIALIZE
    _path = folderName;
    if (!bfs::exists(_path)) {
      bfs::create_directory(_path);
    }
#endif
}

void CCTagFileDebug::newSession(const std::string& sessionName)
{
#ifdef CCTAG_SERIALIZE
    // Don't erase old sessions
    Sessions::iterator it = _sessions.find(sessionName);
    if (it == _sessions.end()) {
        _sstream = &_sessions[sessionName];
    } else {
        _sstream = it->second;
    }
#endif
}

void CCTagFileDebug::outputFlowComponentAssemblingInfos(int status)
{
#if defined CCTAG_SERIALIZE && defined DEBUG
    boost::archive::text_oarchive oa(*_sstream);

    for(const int index : _vflowComponentIndex) {
        oa & BOOST_SERIALIZATION_NVP(index);
    }
    oa & BOOST_SERIALIZATION_NVP(status);
    const bool aux = _isAssembled;
    oa & BOOST_SERIALIZATION_NVP(aux);
    cctag::serializeEllipse(oa, _researchArea);
    _researchArea = cctag::numerical::geometry::Ellipse();
#endif
}

void CCTagFileDebug::printInfos()
{
    CCTAG_COUT("Print infos");
    for(const int index : _vflowComponentIndex) {
        CCTAG_COUT_VAR(index);
    }
    CCTAG_COUT_VAR(_isAssembled);
    CCTAG_PAUSE;
}

void CCTagFileDebug::initFlowComponentsIndex(int size)
{
#if defined CCTAG_SERIALIZE && defined DEBUG
    _vflowComponentIndex.resize(size);
    for (int i = 0; i < _vflowComponentIndex.size(); ++i) {
        _vflowComponentIndex[i] = 0;
    }
#endif            
}

void CCTagFileDebug::resetFlowComponent()
{
#if defined CCTAG_SERIALIZE && defined DEBUG
    _isAssembled = false;
    _researchArea = cctag::numerical::geometry::Ellipse();

    if (_vflowComponentIndex.size() > 1) {
        for (int i = 1; i < _vflowComponentIndex.size(); ++i) {
            _vflowComponentIndex[i] = 0;
        }
    }
#endif            
}

void CCTagFileDebug::incrementFlowComponentIndex(int n)
{
#if defined CCTAG_SERIALIZE && defined DEBUG
    (_vflowComponentIndex[n])++;
#endif            
}

void CCTagFileDebug::setResearchArea(const cctag::numerical::geometry::Ellipse& circularResearchArea)
{
#if defined CCTAG_SERIALIZE && defined DEBUG
    _researchArea = circularResearchArea;
#endif 
}

void CCTagFileDebug::setFlowComponentAssemblingState(bool isAssembled, int indexSelectedFlowComponent)
{
#if defined CCTAG_SERIALIZE && defined DEBUG
    _isAssembled = isAssembled;
    _vflowComponentIndex[1] = indexSelectedFlowComponent;
#endif
}

void CCTagFileDebug::outputFlowComponentInfos(const cctag::CCTagFlowComponent & flowComponent)
{
#ifdef CCTAG_SERIALIZE
    if (_sstream) {
        boost::archive::text_oarchive oa(*_sstream);
        //oa << flowComponent;
        const std::size_t nCircles = flowComponent._nCircles;
        oa & BOOST_SERIALIZATION_NVP(nCircles);
        cctag::serializeFlowComponent(oa, flowComponent);
    } else {
        CCTAG_COUT_ERROR("Unable to output flowComponent infos! Select session before!");
    }
#endif
}

void CCTagFileDebug::outputMarkerInfos(const cctag::CCTag& marker)
{
#ifdef CCTAG_SERIALIZE
    if (_sstream) {
        boost::archive::text_oarchive oa(*_sstream);
        oa << marker;
    } else {
        CCTAG_COUT_ERROR("Unable to output marker infos! Select session before!");
    }
#endif
}

void CCTagFileDebug::outPutAllSessions() const
{
#ifdef CCTAG_SERIALIZE
    for (Sessions::const_iterator it = _sessions.begin(), itEnd = _sessions.end(); it != itEnd; ++it) {
        const std::string filename = _path + "/" + it->first; //cctagFileDebug_
        std::ofstream f(filename.c_str());
        f << it->second->str();
    }
#endif
}

void CCTagFileDebug::clearSessions()
{
#ifdef CCTAG_SERIALIZE
    _sessions.erase(_sessions.begin(), _sessions.end());
#endif
}

// Vote debug
void CCTagFileDebug::newVote(float x, float y, float dx, float dy)
{
#if defined(CCTAG_SERIALIZE) && defined(CCTAG_VOTE_DEBUG)
   if (_sstream) {
      //boost::archive::text_oarchive oa(*_sstream);
      *_sstream << x << " " << y << " " << dx << " " << dy;
  } else {
      CCTAG_COUT_ERROR("Unable to output vote infos!");
  }
#endif
}

void CCTagFileDebug::addFieldLinePoint(float x, float y)
{
#if defined(CCTAG_SERIALIZE) && defined(CCTAG_VOTE_DEBUG)
   if (_sstream) {
      //boost::archive::text_oarchive oa(*_sstream);
      *_sstream << " " << x << " " << y;
  } else {
      CCTAG_COUT_ERROR("Unable to output vote infos!");
  }
#endif
}

void CCTagFileDebug::endVote()
{
#if defined(CCTAG_SERIALIZE) && defined(CCTAG_VOTE_DEBUG)
   if (_sstream) {
      //boost::archive::text_oarchive oa(*_sstream);
      *_sstream << "\n";
  } else {
      CCTAG_COUT_ERROR("Unable to output vote infos!");
  }
#endif
}

} // namespace cctag

