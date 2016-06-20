#ifndef _CCTAG_MARKERS_TYPES_HPP_
#define _CCTAG_MARKERS_TYPES_HPP_

#include <memory>
#include <new>
#include <stdexcept>
#include <cctag/EdgePoint.hpp>


namespace cctag {

class EdgePointCollection
{
#ifndef WITH_CUDA
  static constexpr size_t MAX_POINTS = size_t(1) << 20;
  static constexpr size_t MAX_RESOLUTION = 4096; // 2048;
  static constexpr size_t CUDA_OFFSET = 1024; // 4 kB, one page
  static constexpr size_t MAX_VOTERLIST_SIZE = 16*MAX_POINTS;
#else
  static const size_t MAX_POINTS;
  static const size_t MAX_RESOLUTION;
  static const size_t CUDA_OFFSET;
  static const size_t MAX_VOTERLIST_SIZE;
#endif
  
public:
#ifndef WITH_CUDA
  using int_vector = std::vector<int>;
  using voter_list = std::pair<const int*, const int*>;
#else
  typedef std::vector<int>                  int_vector;
  typedef std::pair<const int*, const int*> voter_list;
#endif
  
private:
  // These must be exported also by CUDA.
#ifndef WITH_CUDA
  std::unique_ptr<int[]> _edgeMap;
  std::unique_ptr<EdgePoint[]> _edgeList;
  std::unique_ptr<int[]> _linkList;     // even idx: before, odd: after
  std::unique_ptr<int[]> _votersIndex;  // with CUDA_OFFSET; [0] is point count
  std::unique_ptr<int[]> _votersList;
#else
  int* _edgeMap;
  EdgePoint* _edgeList;
  int* _linkList;     // even idx: before, odd: after
  int* _votersIndex;  // with CUDA_OFFSET; [0] is point count
  int* _votersList;
#endif
  
  // These are used only on the CPU.
#ifndef WITH_CUDA
  std::unique_ptr<unsigned[]> _processedIn;
  std::unique_ptr<unsigned[]> _processedAux;
#else
  unsigned* _processedIn;
  unsigned* _processedAux;
#endif
  size_t _edgeMapShape[2];
  
#ifndef WITH_CUDA
  static_assert(sizeof(unsigned) == 4, "unsigned has wrong size");
#endif
  
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
#ifndef WITH_CUDA
  EdgePointCollection() = default;
  
  EdgePointCollection(const EdgePointCollection&) = delete;
  
  EdgePointCollection& operator=(const EdgePointCollection&) = delete;
#else
private:
  EdgePointCollection();
  EdgePointCollection(const EdgePointCollection&);
  EdgePointCollection& operator=(const EdgePointCollection&);
public:
#endif
  
  EdgePointCollection(size_t w, size_t h);
    
  void add_point(int vx, int vy, float vdx, float vdy);
  
  int get_point_count()
  {
    return point_count();
  }
    
  const size_t* shape() const { return _edgeMapShape; }
  
#ifndef WITH_CUDA
  EdgePoint* operator()(int i) { return i >= 0 ? &_edgeList[i] : nullptr; }

  EdgePoint* operator()(int i) const { return i >= 0 ? const_cast<EdgePoint*>(&_edgeList[i]) : nullptr; }
#else
  EdgePoint* operator()(int i) { return i >= 0 ? &_edgeList[i] : 0; }
  EdgePoint* operator()(int i) const { return i >= 0 ? const_cast<EdgePoint*>(&_edgeList[i]) : 0; }
#endif

  EdgePoint* operator()(int x, int y) const { return (*this)(_edgeMap[map_index(x,y)]); }

  int operator()(const EdgePoint* p) const
  {
    if (!p)
      return -1;
    if (p < &_edgeList[0] || p >= &_edgeList[point_count()])
      throw std::logic_error("EdgePointCollection::index: invalid pointer");
    return p - &_edgeList[0];
  }

#ifndef WITH_CUDA
  void create_voter_lists(const std::vector<std::vector<int>>& voter_lists);
#else
  void create_voter_lists(const std::vector<std::vector<int> >& voter_lists);
#endif

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

