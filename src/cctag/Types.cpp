
#include <cctag/Types.hpp>

namespace cctag
{

// The input is suboptimal but we don't care: it matters only for the CPU version;
// CUDA version will directly create the required representation.
void EdgePointCollection::create_voter_lists(const std::vector<std::vector<int>>& voter_lists)
{
  if (voter_lists.size() != _edgeList.size())
    throw std::logic_error("EdgePointCollection::create_voters_lists: inconsistent sizes");
  
  // Create index so that list for point i begins at index(i) and ends at index(i+1)
  const size_t n = _edgeList.size();
  _votersIndex.resize(n+1);
  
  _votersIndex[0] = 0;
  for (size_t i = 0; i < n; ++i)
    _votersIndex[i+1] = (int)(_votersIndex[i] + voter_lists[i].size());
  
  _votersList.resize(_votersIndex.back());
  int *p = _votersList.data();
  for (const auto& vlist: voter_lists)
    p = std::copy(vlist.begin(), vlist.end(), p);

  if (p != _votersList.data() + _votersList.size())
    throw std::logic_error("EdgePointCollection::create_voters_lists: invalid count copied");
}

} // namespace cctag
