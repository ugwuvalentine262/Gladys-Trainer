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

#define EDGE_COUNT 7
#define ATTRIBUTE_COUNT 22
#define NODE_COUNT 110

struct board
{
	char mailbox[64];
	int ep_file=0;
	bool friend_has_oo;
	bool friend_has_ooo;
	bool enemy_has_oo;
	bool enemy_has_ooo;

public:

	board(const std::string& fen);

};

struct descriptor
{
	friend class backward_pass;
	friend class forward_pass;

private:

	const board b;
    move moves_[128];
    const int mcount_;
	nn_uint16_t attributes_data[NODE_COUNT][4];
	nn_relation_t relations[NODE_COUNT][8];
	nn_pair_t pairs[EDGE_COUNT][300];
	nn_edge_t edges[EDGE_COUNT];
	nn_node_t nodes[NODE_COUNT];

	nn_attributes_t attributes[NODE_COUNT];
	nn_graphnet_t graphnet;

	int ep_file_;

public:

	descriptor(const board& brd, const move moves[], int mcount);

};

#endif // BOARD_DESCRIPTOR_HPP