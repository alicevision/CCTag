#ifndef _CCTAG_CANDIDATE_HPP_
#define _CCTAG_CANDIDATE_HPP_

#include <cctag/modeConfig.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <boost/unordered/unordered_set.hpp>

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
        
        
#ifdef CCTAG_STAT_DEBUG
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
