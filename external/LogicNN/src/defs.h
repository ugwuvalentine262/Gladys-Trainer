/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef DEFS_H
#define DEFS_H

#include <stdint.h>

#ifdef LNN_ALIGN
    #if defined(LNN_CACHELINE)
        #if LNN_CACHELINE < 0
            #error "LNN_CACHELINE must be positive"
        #elif (LNN_CACHELINE & (LNN_CACHELINE - 1)) != 0
            #error "LNN_CACHELINE must be a power of 2"
        #endif
        #define CACHELINE LNN_CACHELINE
    #else
        #ifndef LNN_CACHELINE
            #define CACHELINE 64
        #else 
            #define CACHELINE 0
        #endif // LNN_CACHELINE
    #endif // defined(LNN_CACHELINE)
#else
    #define CACHELINE 0
#endif // LNN_ALIGN

#if CACHELINE
    #define MAKE_ALIGNED(p, Type) (Type)(((uintptr_t)(p) + CACHELINE - 1) & ~(CACHELINE - 1))
#else
    #define MAKE_ALIGNED(p, Type) (Type)(p)
#endif // CACHELINE

#endif // DEFS_H