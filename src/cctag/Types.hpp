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
public:
  using int_vector = std::vector<int>;
  using voter_list = std::pair<const int*, const int*>;
  
private:
  std::vector<EdgePoint> _edgeList;
  boost::multi_array<int,2> _edgeMap;
  int_vector _voterLists;
  EdgePoint *_currentVoteAdd;
  
public:
  EdgePointCollection()
  {
    _edgeList.reserve(2 << 20);
    _voterLists.reserve(6 << 20);
    _currentVoteAdd = nullptr;
  }
  
  ~EdgePointCollection()
  {
    if (_currentVoteAdd)
      throw std::logic_error("EdgePointCollection: destructing while adding votes");
  }
  
  // Single-argument: returns edge point by its index
  EdgePoint* operator()(int i) { return i >= 0 ? &_edgeList.at(i) : nullptr; }
  const EdgePoint* operator()(int i) const { return i >= 0 ? &_edgeList.at(i) : nullptr; }

  // Two-argument: returns edge point index from the 2D map. MAY RETURN -1!
  int operator()(int i, int j) const
  {
    // XXX: range-check!?
    return _edgeMap[i][j];
  }
  
  // Return the shape of the 2D map
  auto shape() const -> decltype(_edgeMap.shape()) { return _edgeMap.shape(); }

  // Used by CPU voting.
  voter_list voters(const EdgePoint& p) const
  {
    if ((p._votersBegin < 0) != (p._votersEnd < 0))
      throw std::logic_error("EdgePointCollection: invalid voter list indices (1)");
    if (p._votersBegin < 0)
      return std::make_pair(nullptr, nullptr);
    if (p._votersBegin >= (int)_voterLists.size() || p._votersEnd > (int)_voterLists.size())
      throw std::logic_error("EdgePointCollection: invalid voter list indices (2)");
    if (p._votersBegin > p._votersEnd)
      throw std::logic_error("EdgePointCollection: invalid voter list indices (3)");
    return std::make_pair(_voterLists.data() + p._votersBegin, _voterLists.data() + p._votersEnd);
  }
  
  // For CPU implementation of voting
  void begin_add_votes(EdgePoint& p)
  {
    if (_currentVoteAdd)
      throw std::logic_error("EdgePointCollection::begin_add_votes: already in progress");
    _currentVoteAdd = &p;
    p._votersBegin = _voterLists.size();
  }
  void end_add_votes(EdgePoint& p)
  {
    if (&p != _currentVoteAdd)
      throw std::logic_error("EdgePointCollection::end_add_votes: invalid point");
    _currentVoteAdd = nullptr;
    p._votersEnd = _voterLists.size();
  }
  void add_vote(EdgePoint& p, int i)
  {
    if (&p != _currentVoteAdd)
      throw std::logic_error("EdgePointCollection::add_vote: invalid point");
    _voterLists.push_back(i);
  }
};

} // namespace cctag

#endif

