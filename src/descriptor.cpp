/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <sstream>

#include <descriptor.hpp>

#define INSERT_RELATION(rel, rel_inv, node_a, node_b) nn_insert_relation(edges+rel, edges+rel_inv, nodes+node_a, nodes+node_b)
#define INSERT_ATTRIBUTE(attr, node_id) nn_insert_attribute(attr, node_id, attributes)

enum node_type : nn_uint_t 
{
        S01, S02, S03, S04, S05, S06, S07, S08
    ,   S09, S10, S11, S12, S13, S14, S15, S16
    ,   S17, S18, S19, S20, S21, S22, S23, S24
    ,   S25, S26, S27, S28, S29, S30, S31, S32
    ,   S33, S34, S35, S36, S37, S38, S39, S40
    ,   S41, S42, S43, S44, S45, S46, S47, S48
    ,   S49, S50, S51, S52, S53, S54, S55, S56
    ,   S57, S58, S59, S60, S61, S62, S63, S64

    ,   R1, R2, R3, R4, R5, R6, R7, R8
    ,   C1, C2, C3, C4, C5, C6, C7, C8

    ,   D01, D02, D03, D04, D05, D06, D07, D08, D09, D10
    ,   D11, D12, D13, D14, D15, D16, D17, D18, D19, D20
    ,   D21, D22, D23, D24, D25, D26, D27, D28, D29, D30
};

template<class T>
inline T yield(std::stringstream& ss)
{
    T value; ss >> value;
    return value;
}

board::board(const std::string& fen)
    :   mailbox {}
    ,   ep_file(-1)
    ,   friend_has_oo(0)
    ,   friend_has_ooo(0)
    ,   enemy_has_oo(0)
    ,   enemy_has_ooo(0)
{
    bool iw;
    char ch;
    std::stringstream ss(fen);

    ss >> std::noskipws;

    for (auto sq = 56; (ss >> ch) && !isspace(ch); )
    {
        if (isdigit(ch) && ch != '0')  
        {
            sq += (int)(ch - '0');
        }
        else if (ch  ==  '/') {

            sq -= 16;
        }

        else  {

            if (ch=='P') mailbox[sq]=0x1;
            if (ch=='N') mailbox[sq]=0x2;
            if (ch=='B') mailbox[sq]=0x3;
            if (ch=='R') mailbox[sq]=0x4;
            if (ch=='Q') mailbox[sq]=0x5;
            if (ch=='K') mailbox[sq]=0x6;
            if (ch=='p') mailbox[sq]=0x8;
            if (ch=='n') mailbox[sq]=0x9;
            if (ch=='b') mailbox[sq]=0xa;
            if (ch=='r') mailbox[sq]=0xb;
            if (ch=='q') mailbox[sq]=0xc;
            if (ch=='k') mailbox[sq]=0xd;

            sq++;
        }
    }

    iw=yield<char>(ss)=='b'?false:true;

    yield<char>(ss);

    while ((ss >> ch) && !isspace(ch)) 
    {
        if (ch == 'K') friend_has_oo=true;
        if (ch == 'Q') friend_has_ooo=true;
        if (ch == 'k') enemy_has_oo=true;
        if (ch == 'q') enemy_has_ooo=true;
    }

    if ((ss >> ch) && isalpha(ch))
    {
        ep_file=(int)(ch-'a'), yield<char>(ss);
    }

    if (!iw) {
        std::swap(friend_has_ooo, enemy_has_ooo);
        std::swap(friend_has_oo, enemy_has_oo);

        for (int i=0; i < 32; i++) {
            std::swap(mailbox[i], mailbox[i^56]);
        }
    }
}

descriptor::descriptor(const board& brd)
    :   b(brd)
    ,   attributes_data {}
    ,   relations {}
    ,   pairs {}
    ,   edges {}
    ,   nodes {}
    ,   attributes {}
    ,   graphnet {
                .edges=edges
            ,   .nodes=nodes
            ,   .node_count=NODE_COUNT
            ,   .edge_count=EDGE_COUNT
        }
{
    // setup nodes
    for (nn_uint16_t i=0; i < NODE_COUNT; i++)
    {
        attributes[i].attributes=attributes_data[i];
        attributes[i].attributes_count=0;

        nodes[i].relations=relations[i];
        nodes[i].id=i;
    }

    // setup edges
    for (nn_uint16_t i=0; i < EDGE_COUNT; i++)
    {
        edges[i].pairs=pairs[i];
        edges[i].id=i;
    }

    nn_zero_graphnet(&graphnet);
}
