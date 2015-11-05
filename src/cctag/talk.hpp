#pragma once

namespace cctag {
namespace talk {

extern bool on;

} // namespace talk
} // namespace cctag

#ifndef CCTAG_NO_COUT
#define DO_TALK(a) if(cctag::talk::on) { a }
#else // CCTAG_NO_COUT
#define DO_TALK(...)
#endif // CCTAG_NO_COUT
