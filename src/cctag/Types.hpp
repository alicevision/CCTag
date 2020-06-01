/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_MARKERS_TYPES_HPP_
#define _CCTAG_MARKERS_TYPES_HPP_

#include <memory>
#include <new>
#include <stdexcept>
#include <cctag/EdgePoint.hpp>


namespace cctag {

class EdgePointCollection
{
public:
  static constexpr size_t MAX_POINTS = size_t(1) << 24;
private:
  static constexpr size_t MAX_RESOLUTION = 6144;
  static constexpr size_t CUDA_OFFSET = 1024; // 4 kB, one page
  static constexpr size_t MAX_VOTERLIST_SIZE = 16*MAX_POINTS;
  
public:
  using int_vector = std::vector<int>;
  using voter_list = std::pair<const int*, const int*>;
  
private:
  // These must be exported also by CUDA.
  std::unique_ptr<int[]> _edgeMap;
  std::unique_ptr<EdgePoint[]> _edgeList;
  std::unique_ptr<int[]> _linkList;     // even idx: before, odd: after
  std::unique_ptr<int[]> _votersIndex;  // with CUDA_OFFSET; [0] is point count
  std::unique_ptr<int[]> _votersList;
  
  // These are used only on the CPU.
  std::unique_ptr<unsigned[]> _processedIn;
  std::unique_ptr<unsigned[]> _processedAux;
  size_t _edgeMapShape[2]{};
  
  static_assert(sizeof(unsigned) == 4, "unsigned has wrong size");
  
  int& point_count() { return _votersIndex[0]; }
  int point_count() const { return _votersIndex[0]; }
  size_t map_index(int x, int y) const { return x + y * _edgeMapShape[0]; }
  
  static void set_bit(unsigned* v, size_t i, bool f)
  {
    if (i >= MAX_POINTS)
      throw std::out_of_range("EdgePointCollection::set_bit");
    if (f) v[i/4] |=   1U << (i & 31);
    else   v[i/4] &= ~(1U << (i & 31));
  }
  
  static bool test_bit(unsigned* v, size_t i)
  {
    if (i >= MAX_POINTS)
      throw std::out_of_range("EdgePointCollection::test_bit");
    return v[i/4] & (1U << (i & 31));
  }
  
public:
  EdgePointCollection() = default;
  
  EdgePointCollection(const EdgePointCollection&) = delete;
  
  EdgePointCollection& operator=(const EdgePointCollection&) = delete;
  
  EdgePointCollection(size_t w, size_t h);
    
  void add_point(int vx, int vy, float vdx, float vdy);
  
  int get_point_count()
  {
    return point_count();
  }
    
  const size_t* shape() const { return _edgeMapShape; }
  
  EdgePoint* operator()(int i) { return i >= 0 ? &_edgeList[i] : nullptr; }

  EdgePoint* operator()(int i) const { return i >= 0 ? const_cast<EdgePoint*>(&_edgeList[i]) : nullptr; }

  EdgePoint* operator()(int x, int y) const { return (*this)(_edgeMap[map_index(x,y)]); }

  int operator()(const EdgePoint* p) const
  {
    if (!p)
      return -1;
    if (p < &_edgeList[0] || p >= &_edgeList[point_count()])
      throw std::logic_error("EdgePointCollection::index: invalid pointer");
    return p - &_edgeList[0];
  }

  void create_voter_lists(const std::vector<std::vector<int>>& voter_lists);

  voter_list voters(const EdgePoint* p) const
  {
    int i = (*this)(p);
    int b = _votersIndex[i+CUDA_OFFSET], e = _votersIndex[i+1+CUDA_OFFSET];
    return std::make_pair(&_votersList[0]+b, &_votersList[0]+e);
  }
  
  int voters_size(const EdgePoint* p) const
  {
    int i = (*this)(p);
    int b = _votersIndex[i+CUDA_OFFSET], e = _votersIndex[i+1+CUDA_OFFSET];
    return e - b;
  }
  
  EdgePoint* before(EdgePoint* p) const
  {
    int i = (*this)(p);
    return (*this)(_linkList[2*i+0]);
  }

  void set_before(EdgePoint* p, int link)
  {
    int i = (*this)(p);
    _linkList[2*i+0] = link;
  }
  
  EdgePoint* after(EdgePoint* p) const
  {
    int i = (*this)(p);
    return (*this)(_linkList[2*i+1]);
  }

  void set_after(EdgePoint* p, int link)
  {
    int i = (*this)(p);
    _linkList[2*i+1] = link;
  }
  
  void set_processed_in(EdgePoint* p, bool f)
  {
    set_bit(&_processedIn[0], (*this)(p), f);
  }
  
  bool test_processed_in(EdgePoint* p)
  {
    return test_bit(&_processedIn[0], (*this)(p));
  }
  
  void set_processed_aux(EdgePoint* p, bool f)
  {
    set_bit(&_processedAux[0], (*this)(p), f);
  }
  
  bool test_processed_aux(EdgePoint* p)
  {
    return test_bit(&_processedAux[0], (*this)(p));
  }
};

} // namespace cctag

#endif

