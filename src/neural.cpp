/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <Eigen/Dense>

#include <neural.hpp>
#include <board_descriptor.hpp>

static void matmat_mult
    (
            float    const * const W
        ,   float    const * const x
        ,   float          * const y
        ,   unsigned         const batch
        ,   unsigned         const in
        ,   unsigned         const out
    )
{
    
}

static void matTmat_mult
    (
            float    const * const A
        ,   float    const * const B
        ,   float          * const Y
        ,   unsigned         const Acols
        ,   unsigned         const Bcols
        ,   unsigned         const common
    )
{
    
}

static void matmatT_mult
    (
            float    const * const A
        ,   float    const * const B
        ,   float          * const Y
        ,   unsigned         const Arows
        ,   unsigned         const Brows
        ,   unsigned         const common
    )
{
    
}

static void vecmat_mult
    (
            float    const         v[]
        ,   float    const * const W
        ,   float                  y[]
        ,   unsigned const         in
        ,   unsigned const         out
    )
{
    
}

static void softmax
    (
            float          xy[]
        ,   unsigned const dim
    )
{
    
}

static void accumulate
    (
            float          data[]
        ,   float    const delta[]
        ,   unsigned       count
    )
{

}