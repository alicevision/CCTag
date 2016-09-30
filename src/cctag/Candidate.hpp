/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_CANDIDATE_HPP_
#define _CCTAG_CANDIDATE_HPP_

#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <list>

namespace cctag
{
    
class Candidate {
public:
        Candidate(){}
    
	Candidate( EdgePoint* seed, const std::list<EdgePoint*> & convexEdgeSegment,
		const std::vector<EdgePoint*> & outerEllipsePoints, const cctag::numerical::geometry::Ellipse & outerEllipse,
		const std::vector<EdgePoint*> & filteredChildrens, int score, std::size_t nLabel )
		: _seed( seed )
		, _convexEdgeSegment( convexEdgeSegment )
		, _outerEllipsePoints( outerEllipsePoints )
	        , _outerEllipse ( outerEllipse )
		, _filteredChildrens(filteredChildrens)
		, _score(score)
		, _nLabel(nLabel)
	{}

	virtual ~Candidate() {}

	EdgePoint* _seed;
	std::list<EdgePoint*> _convexEdgeSegment;
	std::vector<EdgePoint*> _outerEllipsePoints;
	cctag::numerical::geometry::Ellipse _outerEllipse;
	std::vector<EdgePoint*> _filteredChildrens;
	int _score;
	std::size_t _nLabel;
        float _averageReceivedVote;
        
        
#ifdef CCTAG_SERIALIZE
        // From here -- only used for results analysis --
        std::list<EdgePoint*> _childrens;
        
        void setChildrens(const std::list<EdgePoint*> & childrens){
            _childrens = childrens;
        }        
        
        const std::list<EdgePoint*> & getChildrens(){
            return _childrens;
        }
        
        // To here -- only used for results analysis --
#endif

};

} // namespace cctag

#endif
