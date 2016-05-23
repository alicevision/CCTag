#ifndef _CCTAG_MARKERS_TYPES_HPP_
#define _CCTAG_MARKERS_TYPES_HPP_

#include <stdexcept>
#include <cctag/EdgePoint.hpp>
#include <boost/multi_array.hpp>
#include <boost/unordered/unordered_map.hpp>


namespace cctag {

/**
 * @brief An image (2D array) of pointers to EdgePoints. For each pixel we associate an EdgePoint.
 */
typedef boost::multi_array<EdgePoint*, 2> EdgePointsImage;

// typedef boost::unordered_map< EdgePoint*, std::vector< EdgePoint* > > WinnerMap;  ///< associate a winner with its voters

class EdgePointCollection
{
  using link_pair = std::tuple<int, int>; // 0: before, 1: after
  static_assert(sizeof(link_pair)==8, "int_pair not packed");
  
public:
  using int_vector = std::vector<int>;
  using voter_list = std::pair<const int*, const int*>;
  
private:
  boost::multi_array<int,2> _edgeMap; // XXX: replace with something less magical!
  std::vector<EdgePoint> _edgeList;
  std::vector<link_pair> _linkList;
  int_vector _votersIndex;
  int_vector _votersList;
  
public:
  EdgePointCollection()
  {
    _edgeList.reserve(2 << 20);
    _linkList.reserve(2 << 20);
    _votersIndex.reserve(2 << 20);
    _votersList.reserve(6 << 20);
  }
  
  // General accessors.
  std::vector<EdgePoint>& list() { return _edgeList; }
  const std::vector<EdgePoint>& list() const { return _edgeList; }

  boost::multi_array<int,2>& map() { return _edgeMap; }
  const boost::multi_array<int,2>& map() const { return _edgeMap; }
  
  std::vector<link_pair>& links() { return _linkList; }
  const std::vector<link_pair>& links() const { return _linkList; }

  auto shape() const -> decltype(_edgeMap.shape()) { return _edgeMap.shape(); }
  
  EdgePoint* operator()(int i) { return i >= 0 ? &_edgeList.at(i) : nullptr; }

  EdgePoint* operator()(int i) const { return i >= 0 ? const_cast<EdgePoint*>(&_edgeList.at(i)) : nullptr; }

  EdgePoint* operator()(int i, int j) const { return (*this)(_edgeMap[i][j]); } // XXX@stian: range-check?

  int operator()(const EdgePoint* p) const
  {
    if (!p)
      return -1;
    if (p < _edgeList.data())
      throw std::logic_error("EdgePointCollection::index: invalid pointer (1)");
    int i = p - _edgeList.data();
    if (i >= _edgeList.size())
      throw std::logic_error("EdgePointCollection::index: invalid pointer (2)");
    return i;
  }

  void create_voter_lists(const std::vector<std::vector<int>>& voter_lists);

  voter_list voters(const EdgePoint* p) const
  {
    int i = (*this)(p);
    int b = _votersIndex[i], e = _votersIndex[i+1];
    return std::make_pair(_votersList.data()+b, _votersList.data()+e);
  }
  
  int voters_size(const EdgePoint* p) const
  {
    int i = (*this)(p);
    int b = _votersIndex[i], e = _votersIndex[i+1];
    return e - b;
  }
  
  EdgePoint* before(EdgePoint* p) const
  {
    int i = (*this)(p);
    return (*this)(std::get<0>(_linkList.at(i)));
  }

  void set_before(EdgePoint* p, int link)
  {
    int i = (*this)(p);
    std::get<0>(_linkList.at(i)) = link;
  }
  
  EdgePoint* after(EdgePoint* p) const
  {
    int i = (*this)(p);
    return (*this)(std::get<1>(_linkList.at(i)));
  }

  void set_after(EdgePoint* p, int link)
  {
    int i = (*this)(p);
    std::get<1>(_linkList.at(i)) = link;
  }
};

} // namespace cctag

#endif

