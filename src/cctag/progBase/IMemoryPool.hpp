#ifndef _IMEMORYPOOL_HPP
#define _IMEMORYPOOL_HPP

#include <boost/smart_ptr/intrusive_ptr.hpp>

#include <cstddef>
#include <stdexcept>

namespace rom {

class IUnknown
{
public:
	virtual ~IUnknown()    = 0;
	virtual void addRef()  = 0;
	virtual void release() = 0;
};

class IPoolData : public IUnknown
{
public:
	virtual ~IPoolData ()                     = 0;
	virtual char*        data()               = 0;
	virtual const char*  data() const         = 0;
	virtual const size_t size() const         = 0;
	virtual const size_t reservedSize() const = 0;
};

void intrusive_ptr_add_ref( IPoolData* pData );
void intrusive_ptr_release( IPoolData* pData );

typedef ::boost::intrusive_ptr<IPoolData> IPoolDataPtr;

class IMemoryPool
{
public:
	virtual ~IMemoryPool()                               = 0;
	virtual size_t       getUsedMemorySize() const       = 0;
	virtual std::size_t  getAllocatedAndUnusedMemorySize() const = 0;
	virtual size_t       getAllocatedMemorySize() const  = 0;
	virtual size_t       getAvailableMemorySize() const  = 0;
	virtual size_t       getWastedMemorySize() const     = 0;
	virtual size_t       getMaxMemorySize() const        = 0;
	virtual void         clear( size_t size )            = 0;
	virtual void         clearOne()                      = 0;
	virtual void         clear()                         = 0;
	virtual IPoolDataPtr allocate( const size_t size )   = 0;
	virtual std::size_t  updateMemoryAuthorizedWithRAM() = 0;
};

}

#endif
