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

#define ATTRIBUTE_COUNT 20
#define EDGE_COUNT 8

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

public:

	board_descriptor(const board& brd);

};

#endif // BOARD_DESCRIPTOR_HPP