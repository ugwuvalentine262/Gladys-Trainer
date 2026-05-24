/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <string.h>
#include <logicnn_backprop.h>

static inline nn_void_t nn_package_node_inverse
    (
            nn_node_t  const * const node
        ,   nn_edge_t  const         edges[]
        ,   nn_uint_t  const         indices[]
        ,   nn_float_t const * const dEdK
        ,   nn_float_t const * const dEdM
        ,   nn_float_t       * const y
        ,   nn_uint_t          const dim
    )
{
    for (nn_uint16_t i=0; i < node->relations_count; i++)
    {
        nn_relation_t rel=node->relations[i];
        nn_float_t * dst = y + indices[rel.edge_id];
            
        memcpy(
                dst + dim * rel.pair_idx
            ,   dEdK + i * dim
            ,   dim * sizeof(nn_float_t)
        );

        memcpy(
                dst + dim * (edges[rel.edge_id].pairs_count + rel.pair_idx)
            ,   dEdM + i * dim
            ,   dim * sizeof(nn_float_t)
        );
    }
}

static inline nn_void_t nn_package_edge_inverse
    (
            nn_edge_t     const * const edge
        ,   nn_msg_pass_t const *       mp
        ,   nn_float_t    const * const x
        ,   nn_float_t          * const y
    )
{
    for (nn_uint16_t i=0; i < edge->pairs_count; i++)
    {
        mp->accumulate(
                y + edge->pairs[i].from * mp->dim
            ,   x + i * mp->dim
            ,   mp->dim
        );
    }
}

nn_void_t nn_matTmat_mult(
            nn_float_t const * const A
        ,   nn_float_t const * const B
        ,   nn_float_t       * const Y
        ,   nn_uint_t          const Acols
        ,   nn_uint_t          const Bcols
        ,   nn_uint_t          const common
    )
{
    for (nn_uint_t i=0; i <Bcols; i++) 
    {
        nn_float_t const *g=B+i*common;

        for (nn_uint_t j=0; j <Acols; j++)
        {
            nn_float_t s=0.0f;
            nn_float_t const *h=A+j*common;

            for (nn_uint_t k=0; k < common; k++) 
            { 
                s += h[k] * g[k]; 
            }
            Y[i*Acols+j] = s;
        }
    }
}

nn_void_t nn_matmatT_mult(
            nn_float_t const * const A
        ,   nn_float_t const * const B
        ,   nn_float_t       * const Y
        ,   nn_uint_t          const Arows
        ,   nn_uint_t          const Brows
        ,   nn_uint_t          const common
    )
{
    memset(Y, 0x00u, sizeof(nn_float_t) * Arows * Brows);

    for (nn_uint_t i=0; i < common; i++)
    {
        nn_float_t const *a=A+i*Arows;
        nn_float_t const *b=B+i*Brows;

        for (nn_uint_t j=0; j < Brows; j++)
        {
            nn_float_t *y=Y+j*Arows;

            for (nn_uint_t k=0; k < Arows; k++)
            {
                y[k] += a[k] * b[j];
            }
        }
    }
}

nn_void_t nn_softmax_backprop(
            nn_float_t       dEdxy[]
        ,   nn_float_t const y[]
        ,   nn_uint_t  const dim
    )
{
    nn_float_t dot=0.0f;
    
    for (nn_uint_t i=0; i < dim; i++)
    {
        dot += y[i] * dEdxy[i];
    }

    for (nn_uint_t i=0; i<dim; i++) 
    {
        dEdxy[i] = y[i] * (dEdxy[i] - dot);
    }
}

nn_void_t nn_embedding_backprop(
            nn_embedding_t  const * const emb
        ,   nn_attributes_t         const x[]
        ,   nn_float_t              const dEdy[]
        ,   nn_float_t                    grad[]
        ,   nn_uint_t               const nodes
    )
{
    memset(grad, 0x00u, sizeof(nn_float_t) * emb->attribute_count * emb->dim);

    for(nn_uint_t i=0; i < nodes; i++)
    {
        nn_float_t const *r = dEdy + emb->dim * i;

        for (nn_uint_t j=0; j < x[i].attributes_count; j++)
        {
            emb->accumulate(
                    grad + emb->dim * x[i].attributes[j]
                ,   r
                ,   emb->dim
            );   
        }
    }
}

static nn_void_t nn_msg_computer_backprop
    (
            nn_msg_pass_backprop_t const * const mpbp
        ,   nn_msg_pass_t          const * const mp
        ,   nn_graphnet_t          const * const gn
        ,   nn_uint_t                      const edge_id
        ,   nn_float_t                           dEdx[]
        ,   nn_float_t                           dEdA[]
        ,   nn_float_t                           dEdW[]
        ,   nn_uint_t              const         indices[]
        ,   nn_float_t             const * const X
        ,   nn_float_t                   * const temp
    )
{
    nn_uint_t i=edge_id;
    nn_edge_t const * edge=gn->edges + i;
    nn_uint_t D=edge->pairs_count;
    nn_uint_t d=mp->dim;

    nn_float_t * dLdK=temp + indices[i];
    nn_float_t * dLdM=dLdK + d * D;
    nn_float_t * dLdX=dLdM + d * D;
    nn_float_t * dLdA=dEdA + i * d * d;
    nn_float_t * dLdW=dEdW + i * d * d;

    nn_float_t const *x=X + indices[i] + D * d * 2;
    nn_float_t const *pA=mp->attention + d * d * i;
    nn_float_t const *pW=mp->weights + d * d * i;

    mpbp->matmatT_mult(
            dLdK
        ,   x
        ,   dLdA
        ,   d
        ,   d
        ,   D
    );

    mpbp->matmatT_mult(
            dLdM
        ,   x
        ,   dLdW
        ,   d
        ,   d
        ,   D
    );

    mpbp->matTmat_mult(
            pA
        ,   dLdK
        ,   dLdX
        ,   d
        ,   D
        ,   d
    );

    nn_package_edge_inverse(
            edge
        ,   mp
        ,   dLdX
        ,   dEdx
    );

    mpbp->matTmat_mult(
            pW
        ,   dLdM
        ,   dLdX
        ,   d
        ,   D
        ,   d
    );

    nn_package_edge_inverse(
            edge
        ,   mp
        ,   dLdX
        ,   dEdx
    );
}

static nn_void_t nn_msg_aggregator_backprop
    (
            nn_msg_pass_backprop_t const * const mpbp
        ,   nn_msg_pass_t          const * const mp
        ,   nn_graphnet_t          const * const gn
        ,   nn_float_t             const * const X
        ,   nn_uint_t                      const node_id
        ,   nn_float_t                           dEdx[]
        ,   nn_float_t             const         dEdy[]
        ,   nn_uint_t              const         indices[]
        ,   nn_float_t             const * const messages
        ,   nn_float_t                   * const temp
    )
{
    nn_uint_t i= node_id;
    nn_node_t const *node=gn->nodes + i;
    nn_uint_t N=node->relations_count;
    nn_uint_t d= mp->dim;
    nn_uint_t idx=indices[i+gn->edge_count];

    nn_float_t const *nE= X + i * d;
    nn_float_t const *K= messages + idx;
    nn_float_t const *M= K + d * N;
    nn_float_t const *e= M + d * N;
    nn_float_t const *a= e;

    nn_float_t const *dLdy = dEdy + i * d;
    nn_float_t const *dLdm = dLdy;

    nn_float_t *dLdK = temp + idx;
    nn_float_t *dLdM = dLdK + d * N;
    nn_float_t *dLde = dLdM + d * N;
    nn_float_t *dLda = dLde;
    nn_float_t *dLdnE= dEdx + i * d;

    mp->vecmat_mult(
            dLdm
        ,   M
        ,   dLda
        ,   d
        ,   N
    );

    mpbp->matmatT_mult(
            dLdm
        ,   a
        ,   dLdM
        ,   d
        ,   N
        ,   1
    );

    mpbp->softmax(
            dLda
        ,   a
        ,   N
    );

    mp->matmat_mult(
            K
        ,   dLde
        ,   dLdnE
        ,   1
        ,   N
        ,   d
    );

    mp->accumulate(
            dLdnE
        ,   dLdy
        ,   d
    );

    mpbp->matmatT_mult(
            nE
        ,   dLde
        ,   dLdK
        ,   d
        ,   N
        ,   1
    );

    nn_package_node_inverse(
            node
        ,   gn->edges
        ,   indices
        ,   dLdK
        ,   dLdM
        ,   temp
        ,   d
    );
}

nn_void_t nn_msg_pass_backprop(
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
    )
{
    nn_uint_t eC =gn->edge_count;
    nn_uint_t nC =gn->node_count;
    nn_uint_t d = mp->dim;
    nn_uint_t *indices=(nn_uint_t*)(data);
    nn_float_t *msgmap=(nn_float_t*)(indices + nC + eC);
    nn_float_t *tmp=(nn_float_t*)(temp);

    memset(dEdA, 0x00u, sizeof(nn_float_t) * d * d * eC);
    memset(dEdW, 0x00u, sizeof(nn_float_t) * d * d * eC);
    memset(dEdx, 0x00u, sizeof(nn_float_t) * 1 * d * nC);

    for (nn_uint_t i=0; i < nC; i++)
    {
        nn_uint_t N = gn->nodes[i].relations_count;

        if (!N) {
            continue;
        }

        nn_msg_aggregator_backprop(

                mpbp
            ,   mp
            ,   gn
            ,   X
            ,   i
            ,   dEdx
            ,   dEdy
            ,   indices
            ,   msgmap
            ,   tmp
        );
    }

    for (nn_uint_t i=0; i < eC; i++)
    {
        nn_uint_t D = gn->edges[i].pairs_count;

        if (!D) {
            continue;
        }

        nn_msg_computer_backprop(

                mpbp
            ,   mp
            ,   gn
            ,   i
            ,   dEdx
            ,   dEdA 
            ,   dEdW
            ,   indices
            ,   msgmap
            ,   tmp
        );
    }
}