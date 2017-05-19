/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cctag/Types.hpp>

namespace cctag
{

EdgePointCollection::EdgePointCollection(size_t w, size_t h) :
  _edgeMap(new int[MAX_RESOLUTION*MAX_RESOLUTION]),
  _edgeList(new EdgePoint[MAX_POINTS]),
  _linkList(new int[2*MAX_POINTS]),
  _votersIndex(new int[MAX_POINTS+CUDA_OFFSET]),
  _votersList(new int[MAX_VOTERLIST_SIZE]),
  _processedIn(new unsigned[MAX_POINTS/4]),
  _processedAux(new unsigned[MAX_POINTS/4])
{
  if (w*h > MAX_RESOLUTION*MAX_RESOLUTION)
    throw std::length_error("EdgePointCollection::set_frame_size: image resolution is too large");

  point_count() = 0;
  _edgeMapShape[0] = w; _edgeMapShape[1] = h;
  memset(&_edgeMap[0], -1, w*h*sizeof(int));  // XXX@stian: unnecessary for CUDA
  
  if (w*h/8+4 < MAX_POINTS) {
    memset(&_processedIn[0], 0, w*h/8+4);     // one bit per pixel + roundoff error
    memset(&_processedAux[0], 0, w*h/8+4);    // ditto.
  }
  else {
    memset(&_processedIn[0], 0, MAX_POINTS);
    memset(&_processedAux[0], 0, MAX_POINTS);
  }
}

void EdgePointCollection::add_point(int vx, int vy, float vdx, float vdy)
{
  if (vx < 0 || vx >= _edgeMapShape[0] || vy < 0 || vy >= _edgeMapShape[1])
    throw std::out_of_range("EdgePointCollection::add_point: coordinate out of range");

  size_t imap = map_index(vx, vy);
  if (_edgeMap[imap] != -1)
    throw std::logic_error("EdgePointCollection::add_point: point already exists");

  // XXX@stian: new() below is technically UB, but the class has no defined dtors
  // so it's safe to re-new it in place w/o calling the dtor firs.
  
  if (point_count() >= MAX_POINTS)
    throw std::logic_error(std::string("EdgePointCollection::add_point: too many edge points (nb points: ") + std::to_string(point_count()) + ", max: " + std::to_string(MAX_POINTS) + ")");
  
  size_t ipoint = point_count()++;
  _edgeMap[imap] = ipoint;
  new (&_edgeList[ipoint]) EdgePoint(vx, vy, vdx, vdy);
  _linkList[2*ipoint+0] = -1;
  _linkList[2*ipoint+1] = -1;
  // voter lists must be constructed afterwards
}


// The input is suboptimal but we don't care: it matters only for the CPU version;
// CUDA version will directly create the required representation.
void EdgePointCollection::create_voter_lists(const std::vector<std::vector<int>>& voter_lists)
{
  if (voter_lists.size() != point_count())
    throw std::length_error("EdgePointCollection::create_voters_lists: inconsistent sizes");
  
  _votersIndex[0+CUDA_OFFSET] = 0;
  for (size_t i = 0; i < point_count(); ++i)
    _votersIndex[i+1+CUDA_OFFSET] = (int)(_votersIndex[i+CUDA_OFFSET] + voter_lists[i].size());
  
  if (_votersIndex[point_count()+CUDA_OFFSET] > MAX_VOTERLIST_SIZE)
    throw std::length_error("EdgePointCollection::create_voters_lists: too many voters");
  
  int *p = &_votersList[0];
  for (const auto& vlist: voter_lists)
    p = std::copy(vlist.begin(), vlist.end(), p);

  if (p != &_votersList[0] + _votersIndex[point_count()+CUDA_OFFSET])
    throw std::logic_error("EdgePointCollection::create_voters_lists: invalid count copied");
}

} // namespace cctag
