/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef BOARD_DESCRIPTOR_HPP
#define BOARD_DESCRIPTOR_HPP

#include <string>

#include <logicnn.h>

#include "policy.hpp"

#define EDGE_COUNT 19
#define ATTRIBUTE_COUNT 23
#define NODE_COUNT 113

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
    ,   FA, FB, FC, FD, FE, FF, FG, FH

    ,   D01, D02, D03, D04, D05, D06, D07, D08, D09, D10
    ,   D11, D12, D13, D14, D15, D16, D17, D18, D19, D20
    ,   D21, D22, D23, D24, D25, D26, D27, D28, D29, D30
    ,   G1 , G2 , G3
};

struct board
{
	char mailbox[64];
	int8_t ep_square;
    int8_t repeated;
    bool white;
	bool friend_has_oo;
	bool friend_has_ooo;
	bool enemy_has_oo;
	bool enemy_has_ooo;

public:

    board()=delete;
	board(const std::string& fen);

};

struct descriptor
{
	friend class backward_pass;
	friend class forward_pass;
    friend class policy_head;

private:

	const board b;
    move moves_[128];
    const int mcount_;
	nn_uint16_t attributes_data[NODE_COUNT][4];
	nn_relation_t relations[NODE_COUNT][64];

    nn_pair_t on_pair[256];
    nn_pair_t on_inv_pair[256];
    nn_pair_t adj_pair[84];
    nn_pair_t above_pair[56];
    nn_pair_t above_inv_pair[56];
    nn_pair_t nxt_pair[56];
    nn_pair_t nxt_inv_pair[56];
    nn_pair_t ell_pair[140];
    nn_pair_t ur_pair[49];
    nn_pair_t ur_inv_pair[49];
    nn_pair_t ul_pair[49];
    nn_pair_t ul_inv_pair[49];
    nn_pair_t hpath_pair[448];
    nn_pair_t vpath_pair[448];
    nn_pair_t dpath_pair[280];
    nn_pair_t kpath_pair[280];
    nn_pair_t main1_pair[64];
    nn_pair_t main2_pair[64];
    nn_pair_t main3_pair[64];

	nn_pair_t *pairs[EDGE_COUNT];
	nn_edge_t edges[EDGE_COUNT];
	nn_node_t nodes[NODE_COUNT];

	nn_attributes_t attributes[NODE_COUNT];
	nn_graphnet_t graphnet;

public:

	descriptor(const board& brd, const move moves[], int mcount);

};

#endif // BOARD_DESCRIPTOR_HPP