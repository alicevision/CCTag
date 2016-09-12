/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Label.hpp>
#include <cctag/Labelizer.hpp>

namespace cctag {

int Label::_counter = 0;

Label::Label( const int id )
	: std::vector<LabelEdgePoint>()
	, _pmax()
{
	if( id == -1 )
	{
		_id = _counter++;
	}
	else
	{
		_id = id;
	}
}

Label::Label( const Label& l )
	: std::vector<LabelEdgePoint>( l )
	, _pmax( l._pmax )
	, _id( l._id )
{}

void Label::merge( Label& with )
{
	for( std::vector<LabelEdgePoint>::iterator it = with.begin(); it != with.end(); ++it )
	{
		this->push_back( *it );
		this->back()._label = this;
	}
}

void Label::resetEdgePointsLabel()
{
	for( std::vector<LabelEdgePoint>::iterator it = this->begin(); it != this->end(); ++it )
	{
		it->_label = this;
	}
}

} // namespace cctag
