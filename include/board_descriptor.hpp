/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef BOARD_DESCRIPTOR_HPP
#define BOARD_DESCRIPTOR_HPP

#include <logicnn.h>

#define EDGE_COUNT 7
#define ATTRIBUTE_COUNT 20
#define NODE_COUNT 25

struct board
{
	char mailbox[64];
	int ep=0;
	bool friend_has_oo;
	bool friend_has_ooo;
	bool enemy_has_oo;
	bool enemy_has_ooo;

public:

};

struct board_descriptor
{
    friend class backward_pass;
    friend class forward_pass;

private:

	nn_uint16_t attributes_data[NODE_COUNT][ATTRIBUTE_COUNT];
	nn_relation_t relations[NODE_COUNT][90];
	nn_pair_t pairs[EDGE_COUNT][90];
	nn_edge_t edges[EDGE_COUNT];
	nn_node_t nodes[NODE_COUNT];

	nn_attributes_t attributes[NODE_COUNT];
	nn_graphnet_t graphnet;

	int ep_file_;

public:

	board_descriptor(const board& brd);

};

#endif // BOARD_DESCRIPTOR_HPP