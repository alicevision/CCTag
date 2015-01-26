#ifndef _ROM_RELEASE_HPP_
#define _ROM_RELEASE_HPP_

/*
 * In release mode, ROM_COUT_*_DEBUG are disabled.
 */

/// @see ROM_COUT
#define ROM_COUT_DEBUG(... )
/// @see  ROM_COUT_INFOS
#define ROM_COUT_INFOS_DEBUG
/// @see  ROM_COUT_INFOS
#define ROM_COUT_WITHINFOS_DEBUG(... )
/// @see ROM_IF_DEBUG
#define ROM_IF_DEBUG(... )

#define ROM_COUT_X_DEBUG( N, ... )
#define ROM_COUT_VAR_DEBUG(... )
#define ROM_COUT_VAR2_DEBUG(... )
#define ROM_COUT_VAR3_DEBUG(... )
#define ROM_COUT_VAR4_DEBUG(... )
#define ROM_COUT_WITHINFOS_DEBUG(... )
#define ROM_COUT_WARNING_DEBUG(... )
#define ROM_COUT_ERROR_DEBUG(... )
#define ROM_COUT_FATALERROR_DEBUG(... )
#define ROM_COUT_EXCEPTION_DEBUG(... )

#endif
