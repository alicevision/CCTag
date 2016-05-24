
#include <cctag/Types.hpp>

namespace cctag
{

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
