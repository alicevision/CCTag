#ifndef _CCTAG_PROGBASE_MEMORYPOOL_HPP_
#define _CCTAG_PROGBASE_MEMORYPOOL_HPP_

#include "IMemoryPool.hpp"

#include <cctag/progBase/pattern/Singleton.hpp>

#include <boost/ptr_container/ptr_list.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_set.hpp>

#include <cmath>
#include <cstddef>
#include <climits>
#include <functional>
#include <list>
#include <map>
#include <numeric>
#include <sstream>


namespace rom {

class PoolData; ///< forward declaration

class IPool
{
public:
	virtual ~IPool()                     = 0;
	virtual void referenced( PoolData* ) = 0;
	virtual void released( PoolData* )   = 0;
};

/**
 * @todo tuttle: virtual destructor or nothing in virtual
 */
class MemoryPool : public IMemoryPool
	, public IPool, public Singleton<MemoryPool>
{
public:
	MemoryPool( const std::size_t maxSize = 0 );
	~MemoryPool();

	IPoolDataPtr allocate( const std::size_t size );
	std::size_t  updateMemoryAuthorizedWithRAM();

	void referenced( PoolData* );
	void released( PoolData* );

	std::size_t getUsedMemorySize() const;
	std::size_t getAllocatedAndUnusedMemorySize() const;
	std::size_t getAllocatedMemorySize() const;
	std::size_t getMaxMemorySize() const;
	std::size_t getAvailableMemorySize() const;
	std::size_t getWastedMemorySize() const;

	void clear( std::size_t size );
	void clear();
	void clearOne();

private:
	typedef boost::unordered_set<PoolData*> DataList;
	boost::ptr_list<PoolData> _allDatas; // the owner
	std::map<char*, PoolData*> _dataMap; // the owner
	DataList _dataUsed;
	DataList _dataUnused;
	std::size_t _memoryAuthorized;
	mutable boost::mutex _mutex;
};

}

#endif
