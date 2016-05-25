#ifndef _CCTAG_MARKERS_TYPES_HPP_
#define _CCTAG_MARKERS_TYPES_HPP_

#include <memory>
#include <new>
#include <stdexcept>
#include <cctag/EdgePoint.hpp>


namespace cctag {

class EdgePointCollection
{
  static constexpr size_t MAX_POINTS = size_t(1) << 20;
  static constexpr size_t MAX_RESOLUTION = 2048;
  static constexpr size_t CUDA_OFFSET = 1024; // 4 kB, one page
  static constexpr size_t MAX_VOTERLIST_SIZE = 16*MAX_POINTS;
  
public:
  using int_vector = std::vector<int>;
  using voter_list = std::pair<const int*, const int*>;
  
private:
  std::unique_ptr<int[]> _edgeMap;
  std::unique_ptr<EdgePoint[]> _edgeList;
  std::unique_ptr<int[]> _linkList;     // even idx: before, odd: after
  std::unique_ptr<int[]> _votersIndex;  // with CUDA_OFFSET; [0] is point count
  std::unique_ptr<int[]> _votersList;
  size_t _edgeMapShape[2];
  
  int& point_count() { return _votersIndex[0]; }
  int point_count() const { return _votersIndex[0]; }
  size_t map_index(int x, int y) const { return x + y * _edgeMapShape[0]; }
  
public:
  EdgePointCollection() = default;
  
  EdgePointCollection(const EdgePointCollection&) = delete;
  
  EdgePointCollection& operator=(const EdgePointCollection&) = delete;
  
  EdgePointCollection(size_t w, size_t h) :
    _edgeMap(new int[MAX_RESOLUTION*MAX_RESOLUTION]),
    _edgeList(new EdgePoint[MAX_POINTS]),
    _linkList(new int[2*MAX_POINTS]),
    _votersIndex(new int[MAX_POINTS+CUDA_OFFSET]),
    _votersList(new int[MAX_VOTERLIST_SIZE])
  {
    if (w > MAX_RESOLUTION || h > MAX_RESOLUTION)
      throw std::length_error("EdgePointCollection::set_frame_size: dimension too large");

    point_count() = 0;
    _edgeMapShape[0] = w; _edgeMapShape[1] = h;
    memset(&_edgeMap[0], -1, w*h*sizeof(int));  // XXX@stian: unnecessary for CUDA
  }
    
  void add_point(int vx, int vy, float vdx, float vdy)
  {
    if (vx < 0 || vx >= _edgeMapShape[0] || vy < 0 || vy >= _edgeMapShape[1])
      throw std::out_of_range("EdgePointCollection::add_point: coordinate out of range");
    
    size_t imap = map_index(vx, vy);
    if (_edgeMap[imap] != -1)
      throw std::logic_error("EdgePointCollection::add_point: point already exists");
    
    // XXX@stian: new() below is technically UB, but the class has no defined dtors
    // so it's safe to re-new it in place w/o calling the dtor firs.
    size_t ipoint = point_count()++;
    _edgeMap[imap] = ipoint;
    new (&_edgeList[ipoint]) EdgePoint(vx, vy, vdx, vdy);
    _linkList[2*ipoint+0] = -1;
    _linkList[2*ipoint+1] = -1;
    // voter lists must be constructed afterwards
  }
  
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
    if (p < _edgeList.get())
      throw std::logic_error("EdgePointCollection::index: invalid pointer (1)");
    int i = p - _edgeList.get();
    if (i >= point_count())
      throw std::logic_error("EdgePointCollection::index: invalid pointer (2)");
    return i;
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
};

} // namespace cctag

#endif

