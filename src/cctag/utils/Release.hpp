/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_RELEASE_HPP_
#define _CCTAG_RELEASE_HPP_

/*
 * In release mode, CCTAG_COUT_*_DEBUG are disabled.
 */

/// @see CCTAG_COUT
#define CCTAG_COUT_DEBUG(... )
/// @see  CCTAG_COUT_INFOS
#define CCTAG_COUT_INFOS_DEBUG
/// @see  CCTAG_COUT_INFOS
#define CCTAG_COUT_WITHINFOS_DEBUG(... )
/// @see CCTAG_IF_DEBUG
#define CCTAG_IF_DEBUG(... )

#define CCTAG_COUT_X_DEBUG( N, ... )
#define CCTAG_COUT_VAR_DEBUG(... )
#define CCTAG_COUT_VAR2_DEBUG(... )
#define CCTAG_COUT_VAR3_DEBUG(... )
#define CCTAG_COUT_VAR4_DEBUG(... )
#define CCTAG_COUT_WITHINFOS_DEBUG(... )
#define CCTAG_COUT_WARNING_DEBUG(... )
#define CCTAG_COUT_ERROR_DEBUG(... )
#define CCTAG_COUT_FATALERROR_DEBUG(... )
#define CCTAG_COUT_EXCEPTION_DEBUG(... )

#define POP_INFO(s)
#define POP_ERROR std::cerr << __FILE__ << ":" << __LINE__ << " ERROR: "
#define POP_ENTER std::cerr << __FILE__ << ":" << __LINE__ << " entering " << __func__ << std::endl
#define POP_LEAVE std::cerr << __FILE__ << ":" << __LINE__ << " leaving " << __func__ << std::endl

#endif
