/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef LOGIC_NEURAL_NET_BACKPROP_H
#define LOGIC_NEURAL_NET_BACKPROP_H

#include "logicnn.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct nn_msg_pass_backprop_t nn_msg_pass_backprop_t;

typedef void (*const nn_matTmat_mult_t) 
        (
                nn_float_t const * const A
            ,   nn_float_t const * const B
            ,   nn_float_t       * const Y
            ,   nn_uint_t          const Acols
            ,   nn_uint_t          const Bcols
            ,   nn_uint_t          const common
        );

typedef void (*const nn_matmatT_mult_t) 
        (
                nn_float_t const * const A
            ,   nn_float_t const * const B
            ,   nn_float_t       * const Y
            ,   nn_uint_t          const Arows
            ,   nn_uint_t          const Brows
            ,   nn_uint_t          const common
        );

typedef void (* const nn_softmax_backprop_t) 
        (
                nn_float_t       dEdxy[]
            ,   nn_float_t const y[]
            ,   nn_uint_t  const dim
        );

struct nn_msg_pass_backprop_t
{
    nn_softmax_backprop_t softmax;
    nn_matTmat_mult_t     matTmat_mult;
    nn_matmatT_mult_t     matmatT_mult;
};

extern nn_void_t nn_msg_pass_backprop
    (
            nn_msg_pass_backprop_t       * const mpbp
        ,   nn_msg_pass_t          const * const mp
        ,   nn_graphnet_t          const * const gn
        ,   nn_float_t             const * const X
        ,   nn_float_t             const         dEdy[]
        ,   nn_float_t                           dEdx[]
        ,   nn_float_t                           dEdW[]
        ,   nn_float_t                           dEdA[]
        ,   nn_byte_t              const * const data
        ,   nn_byte_t                    * const temp
    );

extern nn_void_t nn_matTmat_mult
    (
            nn_float_t const * const A
        ,   nn_float_t const * const B
        ,   nn_float_t       * const Y
        ,   nn_uint_t          const Acols
        ,   nn_uint_t          const Bcols
        ,   nn_uint_t          const common
    );

extern nn_void_t nn_matmatT_mult
    (
            nn_float_t const * const A
        ,   nn_float_t const * const B
        ,   nn_float_t       * const Y
        ,   nn_uint_t          const Arows
        ,   nn_uint_t          const Brows
        ,   nn_uint_t          const common
    );

extern nn_void_t nn_embedding_backprop
    (
            nn_embedding_t  const * const emb
        ,   nn_attributes_t         const x[]
        ,   nn_float_t              const dEdy[]
        ,   nn_float_t                    grad[]
        ,   nn_uint_t               const nodes
    );
    
extern nn_void_t nn_softmax_backprop
    (
            nn_float_t       dEdxy[]
        ,   nn_float_t const y[]
        ,   nn_uint_t  const dim
    );

#ifdef __cplusplus 
}
#endif // __cplusplus

#endif // LOGIC_NEURAL_NET_BACKPROP_H