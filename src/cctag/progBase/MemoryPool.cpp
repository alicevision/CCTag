#include "MemoryPool.hpp"
#include "IMemoryPool.hpp"

#include <cctag/progBase/system/memoryInfo.hpp>
#include <cctag/global.hpp>

#include <boost/ptr_container/ptr_list.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <boost/unordered/detail/fwd.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>


namespace rom {

IPool::~IPool() {}

class PoolData : public IPoolData
{
private:
	PoolData(); ///< No default Ctor
	PoolData( const PoolData& ); ///< No copy Ctor
	friend class MemoryPool;

public:
	PoolData( IPool& pool, const std::size_t size )
		: _pool( pool )
		, _id( _count++ )
		, _reservedSize( size )
		, _size( size )
		, _pData( new char[size] )
		, _refCount( 0 )
	{}

	~PoolData()
	{
		delete [] _pData;
	}

public:
	bool operator==( const PoolData& other ) const
	{
		return _id == other._id;
	}

	void addRef();
	void release();

	char*             data()               { return _pData; }
	const char*       data() const         { return _pData; }
	const std::size_t size() const         { return _size; }
	const std::size_t reservedSize() const { return _reservedSize; }

private:
	static std::size_t _count; ///< unique id generator
	IPool& _pool; ///< ref to the owner pool
	const std::size_t _id; ///< unique id to identify one memory data
	const std::size_t _reservedSize; ///< memory allocated
	std::size_t _size; ///< memory requested
	char* const _pData; ///< own the data
	int _refCount; ///< counter on clients currently using this data
};

void intrusive_ptr_add_ref( IPoolData* pData )
{
	pData->addRef();
}

void intrusive_ptr_release( IPoolData* pData )
{
	pData->release();
}

std::size_t PoolData::_count = 0;

void PoolData::addRef()
{
	if( ++_refCount == 1 )
		_pool.referenced( this );
}

void PoolData::release()
{
	if( --_refCount == 0 )
		_pool.released( this );
}

MemoryPool::MemoryPool( const std::size_t maxSize )
	: _memoryAuthorized( maxSize )
{}

MemoryPool::~MemoryPool()
{
	if( !_dataUsed.empty() )
	{
		ROM_COUT_ERROR( "Error inside memory pool. Some data always mark used at the destruction (nb elements:" << _dataUsed.size() << ")" );
	}
}

void MemoryPool::referenced( PoolData* pData )
{
	boost::mutex::scoped_lock locker( _mutex );
	DataList::iterator it = _dataUnused.find( pData );

	if( it != _dataUnused.end() )
	{
		_dataUnused.erase( it );
	}
	else // a really new data
	{
		_allDatas.push_back( pData );
		_dataMap[pData->data()] = pData;
	}
	_dataUsed.insert( pData );
}

void MemoryPool::released( PoolData* pData )
{
	boost::mutex::scoped_lock locker( _mutex );
	_dataUsed.erase( pData );
	_dataUnused.insert( pData );
}

namespace  {

struct DataFitSize : public std::unary_function<PoolData*, void>
{
	DataFitSize( std::size_t size )
		: _size( size )
		, _bestMatchDiff( ULONG_MAX )
		, _pBestMatch( NULL )
	{}

	void operator()( PoolData* pData )
	{
		const std::size_t dataSize = pData->reservedSize();

		if( _size > dataSize )
			return;
		const std::size_t diff = dataSize - _size;
		if( diff >= _bestMatchDiff )
			return;
		_bestMatchDiff = diff;
		_pBestMatch    = pData;
	}

	PoolData* bestMatch()
	{
		return _pBestMatch;
	}

	private:
		const std::size_t _size;
		std::size_t _bestMatchDiff;
		PoolData* _pBestMatch;
};

}

boost::intrusive_ptr<IPoolData> MemoryPool::allocate( const std::size_t size )
{
	PoolData* pData = NULL;

	{
		boost::mutex::scoped_lock locker( _mutex );
		// checking within unused data
		pData = std::for_each( _dataUnused.begin(), _dataUnused.end(), DataFitSize( size ) ).bestMatch();
	}

	if( pData != NULL )
	{
		pData->_size = size;
		return pData;
	}

	const std::size_t availableSize = getAvailableMemorySize();
	if( size > availableSize )
	{
		std::stringstream s;
		s << "MemoryPool can't allocate size:" << size << " because memorySizeAvailable=" << availableSize;
		BOOST_THROW_EXCEPTION( std::length_error( s.str() ) );
	}
	return new PoolData( *this, size );
}

std::size_t MemoryPool::updateMemoryAuthorizedWithRAM()
{
	_memoryAuthorized = /*getUsedMemorySize() +*/ getMemoryInfo()._totalRam;
	return _memoryAuthorized;
}

namespace  {

std::size_t accumulateReservedSize( const std::size_t& sum, const IPoolData* pData )
{
	return sum + pData->reservedSize();
}

std::size_t accumulateWastedSize( const std::size_t& sum, const IPoolData* pData )
{
	return sum + ( pData->reservedSize() - pData->size() );
}

}

std::size_t MemoryPool::getUsedMemorySize() const
{
	boost::mutex::scoped_lock locker( _mutex );
	return std::accumulate( _dataUsed.begin(), _dataUsed.end(), 0, &accumulateReservedSize );
}

std::size_t MemoryPool::getAllocatedAndUnusedMemorySize() const
{
	boost::mutex::scoped_lock locker( _mutex );
	return std::accumulate( _dataUnused.begin(), _dataUnused.end(), 0, &accumulateReservedSize );
}

std::size_t MemoryPool::getAllocatedMemorySize() const
{
	return getUsedMemorySize() + getAllocatedAndUnusedMemorySize();
}

std::size_t MemoryPool::getMaxMemorySize() const
{
	return _memoryAuthorized;
}

std::size_t MemoryPool::getAvailableMemorySize() const
{
	return getMaxMemorySize() - getUsedMemorySize();
}

std::size_t MemoryPool::getWastedMemorySize() const
{
	boost::mutex::scoped_lock locker( _mutex );
	return std::accumulate( _dataUsed.begin(), _dataUsed.end(), 0, std::ptr_fun( &accumulateWastedSize ) );
}

void MemoryPool::clear( std::size_t size )
{
	/// @todo
}

void MemoryPool::clear()
{
	/// @todo
	boost::mutex::scoped_lock locker( _mutex );
	_dataUnused.clear();
}

void MemoryPool::clearOne()
{
	/// @todo
}

}
