#pragma once

#include <stdlib.h>
#include <inttypes.h>

namespace popart
{

class Frame; // forward decl means cctag/*.cpp need not recompile for frame.h

class TagPipe
{
    Frame* _frame[4];
public:
    TagPipe( );
    void prepframe( const uint32_t pix_w, const uint32_t pix_h );
    void tagframe( unsigned char* pix, uint32_t pix_w, uint32_t pix_h );
    void debug( unsigned char* pix );
};

}; // namespace popart

