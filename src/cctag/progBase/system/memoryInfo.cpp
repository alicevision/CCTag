#include "system.hpp"
#include "memoryInfo.hpp"

#if defined( __WINDOWS__ )
 #include <windows.h>
#elif defined( __LINUX__ )
 #include <sys/sysinfo.h>
#elif (defined( __APPLE__ ) && defined(__MACH__))
 #include <sys/types.h>
 #include <sys/sysctl.h>
 #include <unistd.h>
 #include <mach/message.h>
 #include <mach/host_info.h>
 #include <mach/mach_init.h>
 #include <mach/mach_host.h>
 #include <limits>
#else
 #warning "System unrecognized. Can't find memory infos."
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
#elif (defined( __APPLE__ ) && defined(__MACH__))
    xsw_usage vmusage;
    size_t size = sizeof(vmusage);
    if( sysctlbyname("vm.swapusage", &vmusage, &size, NULL, 0) == 0 )
    {
	    infos._totalSwap = vmusage.xsu_total;
	    infos._freeSwap  = vmusage.xsu_avail;
    }
    else
    {
	    infos._totalSwap =
	    infos._freeSwap  = std::numeric_limits<std::size_t>::max();
    }

    int mib[2];
    int64_t physical_memory;
    size_t length;

    // Get the Physical memory size
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    if( sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0 )
    {
	    infos._totalRam  = physical_memory;
    }
    else
    {
	    infos._totalRam  = std::numeric_limits<std::size_t>::max();
    }

    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics_data_t   vmstat;
    if( host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count) == KERN_SUCCESS )
    {
	    infos._freeRam = physical_memory * getpagesize();
    }
    else
    {
	    infos._freeRam = std::numeric_limits<std::size_t>::max();
    }

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
