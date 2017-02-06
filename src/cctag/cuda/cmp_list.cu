/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "cmp_list.h"

using namespace cctag;

__host__
vote_index_sort::vote_index_sort( const HostEdgeList<TriplePoint>& voters )
    : _voters( voters )
{ }

std::ostream& operator<<( std::ostream& ostr, const int2& v )
{
    ostr << "(" << v.x << "," << v.y << ")";
    return ostr;
}

