/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef LOGIC_NEURAL_NET_H
#define LOGIC_NEURAL_NET_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct nn_graphnet_t   nn_graphnet_t;
typedef struct nn_relation_t   nn_relation_t;
typedef struct nn_pair_t       nn_pair_t;
typedef struct nn_node_t       nn_node_t;
typedef struct nn_edge_t       nn_edge_t;
typedef struct nn_msg_pass_t   nn_msg_pass_t;
typedef struct nn_attributes_t nn_attributes_t;
typedef struct nn_embedding_t  nn_embedding_t;

typedef unsigned char  nn_byte_t;
typedef float          nn_float_t;
typedef unsigned       nn_uint_t;
typedef unsigned short nn_uint16_t;
typedef void           nn_void_t;

typedef void (* const nn_accumulate_t) 
        (
                nn_float_t       data[]
            ,   nn_float_t const delta[]
            ,   nn_uint_t        count
        );

typedef void (* const nn_msg_aggregator_t) 
        (
                nn_msg_pass_t const * const mp
            ,   nn_graphnet_t const * const gn
            ,   nn_uint_t             const node_id
            ,   nn_float_t    const * const X
            ,   nn_float_t    const * const messages
            ,   nn_uint_t     const         indices[]
            ,   nn_float_t          * const temp
        );

typedef void (* const nn_msg_computer_t) 
        (
                nn_msg_pass_t const * const mp  
            ,   nn_graphnet_t const * const gn
            ,   nn_uint_t             const edge_id    
            ,   nn_float_t    const * const X
            ,   nn_float_t          * const temp
        );

typedef void (*const nn_matmat_mult_t) 
        (
                nn_float_t const * const W
            ,   nn_float_t const * const x
            ,   nn_float_t       * const y
            ,   nn_uint_t          const batch
            ,   nn_uint_t          const in
            ,   nn_uint_t          const out
        );

typedef void (*const nn_vecmat_mult_t) 
        (
                nn_float_t const         v[]
            ,   nn_float_t const * const W
            ,   nn_float_t               y[]
            ,   nn_uint_t  const         in
            ,   nn_uint_t  const         out
        );

typedef void (* const nn_softmax_t) 
        (
                nn_float_t       xy[]
            ,   nn_uint_t  const dim
        );

struct nn_relation_t 
{
    nn_uint16_t from;
    nn_uint16_t pair_idx;
    nn_uint16_t edge_id;
};

struct nn_pair_t 
{
    nn_uint16_t from;
    nn_uint16_t to;
};

struct nn_node_t
{
    nn_relation_t *relations;

    nn_uint16_t    id;
    nn_uint16_t    relations_count;
};

struct nn_edge_t
{
    nn_pair_t   *pairs;
    nn_uint16_t  id;
    nn_uint16_t  pairs_count;
};

struct nn_graphnet_t
{
    nn_edge_t * const edges;
    nn_node_t * const nodes;
    nn_uint_t   const node_count;
    nn_uint_t   const edge_count;
};

struct nn_msg_pass_t
{
    nn_msg_computer_t        msg_computer;
    nn_msg_aggregator_t      msg_aggregator;
    nn_accumulate_t          accumulate;
    nn_matmat_mult_t         matmat_mult;
    nn_vecmat_mult_t         vecmat_mult;
    nn_softmax_t             softmax;
    nn_float_t const * const weights;
    nn_float_t const * const attention;
    nn_uint_t          const dim;
    nn_uint_t          const edge_count;
};

struct nn_attributes_t 
{
    nn_uint16_t * attributes;
    nn_uint16_t   attributes_count;
};

struct nn_embedding_t
{
    nn_accumulate_t         accumulate;
    nn_float_t      * const data;
    nn_uint_t         const attribute_count;
    nn_uint_t         const dim;
};

extern nn_void_t nn_zero_attributes(nn_attributes_t[], nn_uint_t count);
extern nn_void_t nn_zero_graphnet(nn_graphnet_t *);

extern nn_uint_t nn_required_bytes
    (
            nn_msg_pass_t const * const mp
        ,   nn_graphnet_t const * const gn
    );

extern nn_void_t nn_insert_attribute
    (
            nn_uint16_t     const attribute
        ,   nn_uint16_t     const node_id
        ,   nn_attributes_t       attributes[]
    );

extern nn_void_t nn_insert_relation
    (
            nn_edge_t * const relation
        ,   nn_edge_t * const relation_inverse
        ,   nn_node_t * const node_A
        ,   nn_node_t * const node_B
    );

extern nn_void_t nn_accumulate
    (
            nn_float_t       data[]
        ,   nn_float_t const delta[]
        ,   nn_uint_t        count
    );

extern nn_void_t nn_embedding
    (
            nn_embedding_t  const * const emb
        ,   nn_attributes_t         const x[]
        ,   nn_float_t                    y[]
        ,   nn_uint_t               const nodes
    );

extern nn_void_t nn_msg_aggregator
    (
            nn_msg_pass_t const * const mp
        ,   nn_graphnet_t const * const gn
        ,   nn_uint_t             const node_id
        ,   nn_float_t    const * const X
        ,   nn_float_t    const * const messages
        ,   nn_uint_t     const         indices[]
        ,   nn_float_t          * const temp
    );

extern nn_void_t nn_msg_computer
    (
            nn_msg_pass_t const * const mp  
        ,   nn_graphnet_t const * const gn
        ,   nn_uint_t             const edge_id    
        ,   nn_float_t    const * const X
        ,   nn_float_t          * const temp
    );

extern nn_void_t nn_msg_pass
    (
            nn_msg_pass_t const * const mp
        ,   nn_graphnet_t const * const gn
        ,   nn_float_t    const         x[]
        ,   nn_float_t                  y[]
        ,   nn_byte_t           * const temp
    );

extern nn_void_t nn_matmat_mult
    (
            nn_float_t const * const W
        ,   nn_float_t const * const x
        ,   nn_float_t       * const y
        ,   nn_uint_t          const batch
        ,   nn_uint_t          const in
        ,   nn_uint_t          const out
    );

extern nn_void_t nn_vecmat_mult
    (
            nn_float_t const         v[]
        ,   nn_float_t const * const W
        ,   nn_float_t               y[]
        ,   nn_uint_t  const         in
        ,   nn_uint_t  const         out
    );

extern nn_void_t nn_softmax
    (
            nn_float_t       xy[]
        ,   nn_uint_t  const dim
    );

#ifdef __cplusplus 
}
#endif // __cplusplus

#endif // LOGIC_NEURAL_NET_H