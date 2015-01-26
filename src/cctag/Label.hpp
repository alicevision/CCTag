#ifndef _ROM_VISION_LABEL_HPP_
#define _ROM_VISION_LABEL_HPP_

#include <cstddef>
#include <vector>

namespace rom {
namespace vision {

class LabelEdgePoint;

class Label : public std::vector<LabelEdgePoint>
{
public:
	Label( const int id = -1 );
	Label( const Label& l );
	virtual ~Label() { }

	void merge( Label& with );
	void resetEdgePointsLabel();

public:
	std::vector<LabelEdgePoint*> _pmax; ///< ??
	std::size_t _id;
	static int _counter;
};

}
}

#endif

