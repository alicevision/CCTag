#include "system.hpp"
#include "memoryInfo.hpp"

#if defined( __WINDOWS__ )
 #include <windows.h>
#elif defined( __LINUX__ )
 #include <sys/sysinfo.h>
#else
 #warning "System unrecognized. Can't found memory infos."
 #include <limits>
#endif

MemoryInfo getMemoryInfo()
{
	MemoryInfo infos;

	#if defined( __WINDOWS__ )
	MEMORYSTATUS memory;
	GlobalMemoryStatus( &memory );

	// memory.dwMemoryLoad;
	infos._totalRam = memory.dwTotalPhys;
	infos._freeRam  = memory.dwAvailPhys;
	//memory.dwTotalPageFile;
	//memory.dwAvailPageFile;
	infos._totalSwap = memory.dwTotalVirtual;
	infos._freeSwap  = memory.dwAvailVirtual;
	#elif defined( __LINUX__ )
	struct sysinfo sys_info;
	sysinfo( &sys_info );

	infos._totalRam = sys_info.totalram * sys_info.mem_unit;
	infos._freeRam  = sys_info.freeram * sys_info.mem_unit;
	//infos._sharedRam = sys_info.sharedram * sys_info.mem_unit;
	//infos._bufferRam = sys_info.bufferram * sys_info.mem_unit;
	infos._totalSwap = sys_info.totalswap * sys_info.mem_unit;
	infos._freeSwap  = sys_info.freeswap * sys_info.mem_unit;
//	TUTTLE_COUT_VAR( sys_info.sharedram * sys_info.mem_unit );
//	TUTTLE_COUT_VAR( sys_info.bufferram * sys_info.mem_unit );
	#else
	infos._totalRam             =
	    infos._freeRam          =
	        infos._totalSwap    =
	            infos._freeSwap = std::numeric_limits<std::size_t>::max();
	#endif

	return infos;
}

std::ostream& operator<<( std::ostream& os, const MemoryInfo& infos )
{
	os << "total ram:" << infos._totalRam << std::endl
	   << "free ram:" << infos._freeRam << std::endl
	   << "total swap:" << infos._totalSwap << std::endl
	   << "free swap:" << infos._freeSwap << std::endl;
	return os;
}
