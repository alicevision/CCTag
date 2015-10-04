#pragma once

namespace cctag {
namespace talk {

extern bool on;

} // namespace talk
} // namespace cctag

#define DO_TALK(a) if(cctag::talk::on) { a }
