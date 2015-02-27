#ifndef _POPART_VISION_LABELIZER_HPP_
#define _POPART_VISION_LABELIZER_HPP_

#include "EdgePoint.hpp"
#include "Label.hpp"

#include <cctag/filter/thinning.hpp>
#include <cctag/global.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/multi_array.hpp>
#include <boost/progress.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>

#include <cstddef>
#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <utility>
#include <vector>

namespace popart {
namespace vision {

struct LabelEdgePoint : public EdgePoint
{
public:
	LabelEdgePoint( const LabelEdgePoint& p )
		: EdgePoint( p )
		, _label( p._label )
	{}

	LabelEdgePoint( const int vx, const int vy, const float vdx, const float vdy, Label * label )
		: EdgePoint( vx, vy, vdx, vdy )
		, _label( label )
	{
	}

	Label *_label;
};

typedef boost::multi_array<LabelEdgePoint*, 2> LabelizedImage;

class Labelizer
{
public:
	Labelizer();
	virtual ~Labelizer();

	template<class SView, class CView, class DXView, class DYView>
	void labelize( SView& svw, CView & cvw, DXView & dx, DYView & dy );

	template<class SView>
	void findLabels( const SView& cvw, const boost::gil::gray16s_view_t& dxVw, const boost::gil::gray16s_view_t& dyVw, std::vector<Label>& labelsList, LabelizedImage& labelsMap );

	void followContour( Label* label, const int x, const int y, const boost::gil::gray16s_view_t& dxVw, const boost::gil::gray16s_view_t& dyVw, LabelizedImage& labelsMap );

	const std::vector<Label>& labels() const    { return _labelsList; }
	std::vector<Label>&       labels()          { return _labelsList; }
	const LabelizedImage&     labelsMap() const { return _labelsMap; }

private:
	LabelizedImage _labelsMap;
	std::vector<Label> _labelsList;
};

template<class SView, class CView, class DXView, class DYView>
void Labelizer::labelize( SView& svw, CView & cvw, DXView & dx, DYView & dy )
{
	using namespace boost::gil;

	boost::timer t;

//	boost::multi_array<popart::vision::LabelEdgePoint*, 2> labelsMap;

	/**************************************************************************
	 *  labelize detected edges and map them with an int id into labelsImage  *
	 **************************************************************************/
	_labelsMap.resize( boost::extents[svw.width()][svw.height()] );
	std::fill( _labelsMap.origin(), _labelsMap.origin() + _labelsMap.size(), (LabelEdgePoint*)NULL );

	_labelsList.push_back( Label( 1 ) );
	Label* label = &_labelsList.back();
	label->reserve( cvw.width() * cvw.height() / 2 ); //TODO évaluer la borne max du nombre de point contour

	/*{*/
	//IplImage* smoothImg = cvCreateImage(cvGetSize(simg), IPL_DEPTH_8U/*img->depth*/, /*img->nChannels*/ 1 );
	//cvSmooth(simg, smoothImg, CV_GAUSSIAN, 9, 0, 1, 0);

	//cvNamedWindow("Smooth", 0);
	//cvShowImage("Smooth", smoothImg);
	//cvWaitKey(0);
	/*}*/

	t.restart();

	/**************************************************************************
	*  Get contours                                                          *
	**************************************************************************/
	{
		//png_write_view("data/pictures/canny.png", cvw);
	}

	t.restart();

	ROM_COUT_LILIAN( "Width : " << cvw.width() << ", Height :" << cvw.height() );
	for( int y = 0 ; y < cvw.height() ; ++y )
	{
		typename CView::x_iterator itc = cvw.row_begin( y );
		typename DXView::x_iterator itDx = dx.row_begin( y );
		typename DYView::x_iterator itDy = dy.row_begin( y );
		for( int x = 0 ; x < cvw.width() ; ++x )
		{
			if( (*itc)[0] )
			{
				label->push_back( LabelEdgePoint( x, y, (*itDx)[0], (*itDy)[0], label ) );
				LabelEdgePoint* p = &label->back();
				p->_label = label;
				_labelsMap[x][y] = p;
			}
			else
			{
				_labelsMap[x][y] = NULL;
			}
			++itc;
			++itDx;
			++itDy;
		}
	}

	ROM_COUT_LILIAN( "Création du label 1 " );

	//findLabels( cvw, dxVw, dyVw, _labelsList, _labelsMap );
	ROM_COUT_LILIAN( "Labelization took: " << t.elapsed() );
	//{
	//	gray8_image_t limg(cvw.dimensions());
	//	gray8_view_t lvw(view(limg));

	//	std::cout << "Width : " << lvw.width() << "Height :" << lvw.height() << std::endl;
	//	for(int x = 0; x < lvw.width(); ++x)
	//	{
	//		for(int y = 0; y < lvw.height(); ++y)
	//		{
	//			if (_labelsMap[x][y])
	//				(*lvw.xy_at(x,y)) = (unsigned char)(_labelsMap[x][y]->_label->_id * 256.0f / _labelsList.size());
	//			else
	//				(*lvw.xy_at(x,y)) = 0;
	//		}
	//	}
	//	png_write_view("data/pictures/labels.png", lvw);
	//}
	//std::cout << "Writing " << std::endl;

	//cvSaveImage("data/output/contour.png", img);
}

template<class SView>
void Labelizer::findLabels( const SView& cvw, const boost::gil::gray16s_view_t& dxVw, const boost::gil::gray16s_view_t& dyVw, std::vector<Label>& labelsVect, LabelizedImage& labelsMap )
{
	int x = 0, y = 0;

	using namespace boost::gil;
	//using namespace popart::numerical;

	typedef std::list<LabelEdgePoint> PointList;
	std::list<PointList> labels;
	int numLabels = 0;
	///@todo replace this with list with range instead of array
	boost::multi_array<PointList*, 2 > lines( boost::extents[cvw.width()][2] );

	std::fill( lines.data(), lines.data() + lines.num_elements(), (PointList*)NULL );

	typedef typename SView::x_iterator sIterator;
	sIterator cannyIt = cvw.row_begin( 0 );
	// Search for labels on the first line
	while( x < cvw.width() )
	{
		while( x < cvw.width() && !*cannyIt )
		{
			++x;
			++cannyIt;
		}
		// If label found
		if( x < cvw.width() )
		{
			labels.push_back( PointList() );
			PointList& label = labels.back();
			++numLabels;
			gray16s_view_t::x_iterator dxit = dxVw.row_begin( y );
			gray16s_view_t::x_iterator dyit = dyVw.row_begin( y );
			dxit += x;
			dyit += x;
			do
			{
				label.push_back( LabelEdgePoint( x, y, *dxit++, *dyit++, NULL ) );
				lines[x][y % 2] = &label;
				++x;
			}
			while( x < cvw.width() && *++cannyIt );
		}
	}

	for( y = 1; y < cvw.height(); ++y )
	{
		cannyIt = cvw.row_begin( y );
		///@todo change & check unordered map
		std::map<PointList*, PointList*> newAssoc;
		x = 0;
		for( int xx = 0; xx < cvw.width(); ++xx )
		{
			lines[xx][y % 2] = NULL;
			if( lines[xx][( y - 1 ) % 2] )
			{
				newAssoc[lines[xx][( y - 1 ) % 2]] = lines[xx][( y - 1 ) % 2];
			}
		}
		///@todo change & check unordered set
		std::map<PointList*, bool> alreadyDone;
		while( x < cvw.width() )
		{
			while( x < cvw.width() && !*cannyIt )
			{
				++x; ++cannyIt;
			}
			// If label found
			if( x < cvw.width() )
			{
				int xx = x;
				// Search for the right label to merge with
				PointList* rightLabel = NULL;
				// Check lower diagonal
				if( x > 0 && lines[x - 1][( y - 1 ) % 2] )
				{
					rightLabel = newAssoc[lines[x - 1][( y - 1 ) % 2]];
				}
				do
				{
					if( lines[x][( y - 1 ) % 2] )
					{
						rightLabel = newAssoc[lines[x][( y - 1 ) % 2]];
					}
					++x;
				}
				while( x < cvw.width() && *++cannyIt );
				// Check higher diagonal
				if( x < cvw.width() && lines[x][( y - 1 ) % 2] )
				{
					rightLabel = newAssoc[lines[x][( y - 1 ) % 2]];
				}

				/////////////// Merging /////////////////

				x       = xx;
				cannyIt = cvw.row_begin( y ) + x;

				// If no adjacent label found, create a new label
				if( !rightLabel )
				{
					labels.push_back( PointList() );
					rightLabel = &labels.back();
					++numLabels;

					gray16s_view_t::x_iterator dxit = dxVw.row_begin( y );
					gray16s_view_t::x_iterator dyit = dyVw.row_begin( y );
					dxit += x;
					dyit += x;

					do
					{
						rightLabel->push_back( LabelEdgePoint( x, y, *dxit++, *dyit++, NULL ) );
						lines[x][y % 2] = rightLabel;
						++x;
					}
					while( x < cvw.width() && *++cannyIt );
				}
				else
				{
					alreadyDone[newAssoc[rightLabel]] = true;
					// Merge with distinct other adjacent labels
					// Check lower diagonal for merge
					if( x > 0 && lines[x - 1][( y - 1 ) % 2] && alreadyDone.find( newAssoc[lines[x - 1][( y - 1 ) % 2]] ) == alreadyDone.end() )
					{
						rightLabel->splice( rightLabel->end(), *newAssoc[lines[x - 1][( y - 1 ) % 2]] );
						newAssoc[lines[x - 1][( y - 1 ) % 2]]->clear();
						--numLabels;
						alreadyDone[newAssoc[lines[x - 1][( y - 1 ) % 2]]] = true;
						newAssoc[lines[x - 1][( y - 1 ) % 2]]              = rightLabel;
					}

					gray16s_view_t::x_iterator dxit = dxVw.row_begin( y );
					gray16s_view_t::x_iterator dyit = dyVw.row_begin( y );
					dxit += x;
					dyit += x;

					do
					{
						// If need to merge
						if( lines[x][( y - 1 ) % 2] && alreadyDone.find( newAssoc[lines[x][( y - 1 ) % 2]] ) == alreadyDone.end() )
						{
							// merge
							rightLabel->splice( rightLabel->end(), *newAssoc[lines[x][( y - 1 ) % 2]] );
							newAssoc[lines[x][( y - 1 ) % 2]]->clear();
							--numLabels;
							alreadyDone[newAssoc[lines[x][( y - 1 ) % 2]]] = true;
							newAssoc[lines[x][( y - 1 ) % 2]]              = rightLabel;
						}
						rightLabel->push_back( LabelEdgePoint( x, y, *dxit++, *dyit++, NULL ) );
						lines[x][y % 2] = rightLabel;
						++x;
					}
					while( x < cvw.width() && *++cannyIt );
					// Check higher diagonal
					if( x < cvw.width() && lines[x][( y - 1 ) % 2] && alreadyDone.find( newAssoc[lines[x][( y - 1 ) % 2]] ) == alreadyDone.end() )
					{
						// append at the end of the right label
						rightLabel->splice( rightLabel->end(), *newAssoc[lines[x][( y - 1 ) % 2]] );
						newAssoc[lines[x][( y - 1 ) % 2]]->clear();
						--numLabels;
						alreadyDone[newAssoc[lines[x][( y - 1 ) % 2]]] = true;
						newAssoc[lines[x][( y - 1 ) % 2]]              = rightLabel;
					}
				}
			}
		}
	}
	labelsVect.reserve( numLabels );
	int lid = 0;
	for( std::list<PointList>::iterator it = labels.begin(); it != labels.end(); ++it )
	{
		if( it->size() > 0 )
		{
			labelsVect.push_back( Label( ++lid ) );
			Label* label = &labelsVect.back();
			label->reserve( it->size() );
			for( PointList::const_iterator itp = it->begin(); itp != it->end(); ++itp )
			{
				label->push_back( *itp );
				LabelEdgePoint* p = &label->back();
				labelsMap[p->x()][p->y()] = p;
				p->_label = label;
			}
		}
	}
}

}
}

#endif
