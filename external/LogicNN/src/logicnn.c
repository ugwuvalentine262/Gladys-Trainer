/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <logicnn.h>

#include "defs.h"

static inline nn_void_t nn_package_node 
    (
            nn_node_t  const * const node
        ,   nn_edge_t  const         edges[]
        ,   nn_float_t const * const x
        ,   nn_uint_t  const         indices[]
        ,   nn_float_t       * const ya
        ,   nn_float_t       * const yw
        ,   nn_uint_t          const dim
    )
{
    for (nn_uint16_t i=0; i < node->relations_count; i++)
    {
        nn_relation_t rel=node->relations[i];
        nn_float_t const*src= x + indices[rel.edge_id];

        memcpy(
                ya + i * dim
            ,   src + rel.pair_idx * dim
            ,   dim * sizeof(nn_float_t)
        );

        memcpy(
                yw + i * dim
            ,   src + dim * (edges[rel.edge_id].pairs_count + rel.pair_idx)
            ,   dim * sizeof(nn_float_t)
        );
    }
}

static inline nn_void_t nn_package_edge
    (
            nn_edge_t  const * const edge
        ,   nn_float_t const * const x
        ,   nn_float_t       * const y
        ,   nn_uint_t          const dim
    )
{
    for (nn_uint16_t i=0; i < edge->pairs_count; i++)
    {
        memcpy(
                y + i * dim
            ,   x + edge->pairs[i].from * dim
            ,   dim * sizeof(nn_float_t)
        );
    }
}

static inline nn_void_t nn_add_relation
    (
            nn_edge_t       * const edge
        ,   nn_node_t       * const to
        ,   nn_node_t const * const from 
    )
{
    nn_relation_t *rel=to->relations + to->relations_count;
    nn_pair_t *pair=edge->pairs + edge->pairs_count;

    rel->from=from->id;
    rel->pair_idx=edge->pairs_count;
    rel->edge_id=edge->id;

    pair->from=from->id;
    pair->to=to->id;

    to->relations_count++;
    edge->pairs_count++;
}

nn_void_t nn_vecmat_mult(
            nn_float_t const         v[]
        ,   nn_float_t const * const W
        ,   nn_float_t               y[]
        ,   nn_uint_t  const         in
        ,   nn_uint_t  const         out
    )
{
    memset(y, 0x00u, sizeof(nn_float_t) * out);

    for (nn_uint_t i=0; i < out; i++) 
    {
        const nn_float_t *w = W + i * in;

        for (nn_uint_t j=0; j < in; j++) 
        { 
            y[i] += v[j] * w[j]; 
        }
    }
}

nn_void_t nn_matmat_mult(
            nn_float_t const * const W
        ,   nn_float_t const * const x
        ,   nn_float_t       * const y
        ,   nn_uint_t          const batch
        ,   nn_uint_t          const in
        ,   nn_uint_t          const out
    )
{
    memset(y, 0x00u, sizeof(nn_float_t) * out * batch);

    for (nn_uint_t i=0; i < batch; i++) 
    {
        nn_float_t *r = y + i * out;
        nn_float_t const *g = x + i * in;

        for (nn_uint_t j=0; j < in; j++)
        {
            nn_float_t const *w = W + j * out;

            for (nn_uint_t k=0; k < out; k++)
            {
                r[k] += w[k] * g[j];
            }
        }
    }
}

nn_void_t nn_softmax(
            nn_float_t       xy[]
        ,   nn_uint_t  const dim
    )
{
    nn_float_t sum=0, max=-1;

    for (nn_uint_t i=0; i < dim; i++) 
    {
        max = xy[i] > max ? xy[i] : max;
    }

    for (nn_uint_t i=0; i < dim; i++) 
    {
        xy[i] = exp(xy[i] - max);
        sum += xy[i];
    }

    for (nn_uint_t i=0; i < dim; i++) 
    {
        xy[i] /= sum;
    }
}

nn_void_t nn_accumulate(
            nn_float_t       data[]
        ,   nn_float_t const delta[]
        ,   nn_uint_t        count
    )
{
    for (nn_uint_t i=0; i < count; i++)
    {
        data[i] += delta[i];
    }
}

nn_void_t nn_embedding(
            nn_embedding_t  const * const emb
        ,   nn_attributes_t         const x[]
        ,   nn_float_t                    y[]
        ,   nn_uint_t               const nodes
    )
{
    memset(y, 0x0, sizeof(nn_float_t) * nodes * emb->dim);

    for (nn_uint_t i=0; i < nodes; i++)
    {
        nn_float_t *r = y + emb->dim * i;

        for (nn_uint_t j=0; j < x[i].attributes_count; j++)
        {
            emb->accumulate(
                    r
                ,   emb->data + emb->dim * x[i].attributes[j]
                ,   emb->dim
            );   
        }
    }
}

nn_void_t nn_msg_aggregator(
            nn_msg_pass_t const * const mp
        ,   nn_graphnet_t const * const gn
        ,   nn_uint_t             const node_id
        ,   nn_float_t    const * const X
        ,   nn_float_t    const * const messages
        ,   nn_uint_t     const         indices[]
        ,   nn_float_t          * const temp
    )
{
    nn_uint_t i=node_id;
    nn_node_t const *node=gn->nodes + i;
    nn_uint_t N=node->relations_count;
    nn_uint_t d=mp->dim;

    nn_float_t const *nE= X + i * d;

    nn_float_t *K= temp;
    nn_float_t *M= K + d * N;
    nn_float_t *y= M + d * N;
    nn_float_t *e= y + d;
    nn_float_t *a= e;

    nn_package_node(
            node
        ,   gn->edges
        ,   messages
        ,   indices
        ,   K
        ,   M
        ,   mp->dim
    );

    mp->vecmat_mult(

            nE
        ,   K
        ,   e
        ,   d
        ,   N
    );

    mp->softmax(e, N);

    mp->matmat_mult(

            M
        ,   a
        ,   y
        ,   1
        ,   N
        ,   d
    );

    mp->accumulate(y, nE, d);
}

nn_void_t nn_msg_computer(
            nn_msg_pass_t const * const mp  
        ,   nn_graphnet_t const * const gn
        ,   nn_uint_t             const edge_id    
        ,   nn_float_t    const * const X
        ,   nn_float_t          * const temp
    )
{
    nn_uint_t i=edge_id;
    nn_edge_t const * edge=gn->edges+i;
    nn_uint_t D=edge->pairs_count;
    nn_uint_t d=mp->dim;

    nn_float_t *yA=temp;
    nn_float_t *yW=yA + d * D;
    nn_float_t *x=yW + d * D;

    nn_float_t const *pA=mp->attention + d * d * i;
    nn_float_t const *pW=mp->weights + d * d * i;

    nn_package_edge(
            edge
        ,   X
        ,   x
        ,   mp->dim
    );

    mp->matmat_mult(
            pA
        ,   x
        ,   yA
        ,   D
        ,   d
        ,   d
    );

    mp->matmat_mult(
            pW
        ,   x
        ,   yW
        ,   D
        ,   d
        ,   d
    );
}

nn_void_t nn_msg_pass(
            nn_msg_pass_t const * const mp
        ,   nn_graphnet_t const * const gn
        ,   nn_float_t    const         x[]
        ,   nn_float_t                  y[]
        ,   nn_byte_t           * const temp
    )
{
    nn_uint_t eC =gn->edge_count;
    nn_uint_t nC =gn->node_count;
    nn_uint_t d = mp->dim;
    nn_uint_t *indices=(nn_uint_t*)(temp);
    nn_float_t *msgmap=MAKE_ALIGNED(indices + eC + nC, nn_float_t *);
    nn_float_t *tmp=msgmap;

    for (nn_uint_t i=0; i < eC; i++)
    {
        nn_uint_t D =gn->edges[i].pairs_count;

        indices[i] = (nn_uint_t)(tmp - msgmap);

        if (!D) {
            continue;
        }

        mp->msg_computer(

                mp
            ,   gn
            ,   i
            ,   x
            ,   tmp
        );

        tmp += D * d * 3;
        tmp = MAKE_ALIGNED(tmp, nn_float_t *);
    }

    for (nn_uint_t i=0; i < nC; i++)
    {
        nn_uint_t N = gn->nodes[i].relations_count;

        indices[i + eC]=(nn_uint_t)(tmp - msgmap);

        if (!N) {
            continue;
        }

        mp->msg_aggregator(

                mp
            ,   gn
            ,   i
            ,   x
            ,   msgmap
            ,   indices
            ,   tmp
        );

        tmp += N * (2 * d + 1) + d;
        tmp = MAKE_ALIGNED(tmp, nn_float_t *);
    }

    for (nn_uint_t i=0; i < nC; i++)
    {
        memcpy(
                y + i * d
            ,   msgmap + indices[i+eC] + gn->nodes[i].relations_count * 2 * d
            ,   d * sizeof(nn_float_t)
        );
    }
}

nn_void_t nn_zero_attributes(nn_attributes_t attributes[], nn_uint_t count)
{
    for (nn_uint_t i=0; i < count; i++)
    {
        attributes[i].attributes_count=0;
    }
}

nn_void_t nn_zero_graphnet(nn_graphnet_t *gn)
{
    for (nn_uint_t i=0; i < gn->edge_count; i++)
    {
        gn->edges[i].pairs_count=0;
    }

    for (nn_uint_t i=0; i < gn->node_count; i++)
    {
        gn->nodes[i].relations_count=0;
    }
}

nn_uint_t nn_required_bytes(
            nn_msg_pass_t const * const mp
        ,   nn_graphnet_t const * const gn
    )
{
    nn_uint_t count=63 + sizeof(nn_uint_t) * (gn->edge_count + gn->node_count);

    count = MAKE_ALIGNED(count, nn_uint_t);

    for (nn_uint_t i=0; i < gn->edge_count; i++) {
        if (gn->edges[i].pairs_count) {
            count += sizeof(nn_float_t) * gn->edges[i].pairs_count * mp->dim * 3;
            count = MAKE_ALIGNED(count, nn_uint_t);
        }
    }

    for (nn_uint_t i=0; i < gn->node_count; i++) {
        if (gn->nodes[i].relations_count) {
            count += sizeof(nn_float_t) * (gn->nodes[i].relations_count * (2 * mp->dim + 1) + mp->dim);
            count = MAKE_ALIGNED(count, nn_uint_t);
        }
    }

    return count;
}

nn_void_t nn_insert_attribute(
            nn_uint16_t     const attribute
        ,   nn_uint16_t     const node_id
        ,   nn_attributes_t       attributes[]
    )
{
    attributes[node_id].attributes[attributes[node_id].attributes_count]=attribute;
    attributes[node_id].attributes_count++;
}

nn_void_t nn_insert_relation(
            nn_edge_t * const relation
        ,   nn_edge_t * const relation_inverse
        ,   nn_node_t * const node_A
        ,   nn_node_t * const node_B
    )
{
    nn_add_relation(
            relation
        ,   node_A
        ,   node_B
    );

    nn_add_relation(
            relation_inverse
        ,   node_B
        ,   node_A
    );
}

nn_void_t nn_insert_link(
            nn_edge_t * const relation
        ,   nn_node_t * const node_to
        ,   nn_node_t * const node_from
    )
{
    nn_add_relation(
            relation
        ,   node_to
        ,   node_from
    );
}