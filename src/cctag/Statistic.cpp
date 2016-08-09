#include <algorithm>
#include "Statistic.hpp"
#include "utils/pcg_random.hpp"

namespace cctag {
namespace numerical {


void rand_5_k(std::array<int, 5>& perm, size_t N)
{
  static thread_local pcg32 rng(271828);
  
  auto it = perm.begin();
  int r;
  
  for (int i = 0; i < 5; ++i) {
    do {
      r = rng(N);
    } while (std::find(perm.begin(), it, r) != it);
    *it++ = r;
  }
}

}
}