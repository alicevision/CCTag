#include <random>
#include <vector>
#include <algorithm>
#include "Statistic.hpp"

#include <cstdlib>

namespace cctag {
namespace numerical {

static constexpr size_t MAX_POINTS = 1000;
static constexpr size_t MAX_RANDOMNESS = 10000000;

struct Randomness : public std::vector<unsigned short> 
{
  Randomness();
};

static Randomness randomness;

Randomness::Randomness() : std::vector<unsigned short>(MAX_RANDOMNESS)
{
  std::ranlux24 engine;
  engine.seed(2718282);
  std::uniform_int_distribution<int> dist(0, MAX_POINTS);
  for (size_t i = 0; i < MAX_RANDOMNESS; ++i)
    randomness[i] = dist(engine);
}

void rand_5_k(std::array<int, 5>& perm, size_t N)
{
  static thread_local int sourceIndex = 0;
  
  auto it = perm.begin();
  int r;
  
  for (int i = 0; i < 5; ++i) {
retry:
    do {
      if (sourceIndex >= MAX_RANDOMNESS) sourceIndex = 0;
      r = randomness[sourceIndex++];
    } while (r >= N);
    if (std::find(perm.begin(), it, r) != it)
      goto retry;
    *it++ = r;
  }
}

}
}