/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <cstring>
#include <algorithm>
#include <sstream>

#include "policy.hpp"

move::move(const std::string& algebra)
{
    auto square = [](char file, char rank)
        {
            if (   file < 'a' || rank < '1' 
                || file > 'h' || rank > '8'
               )
            {
                throw move::bad();
            }
            return (int)(rank - '1') * 8 + (int)(file - 'a');
        };

    if (algebra.size()!=4 && algebra.size()!=5)
    {
        throw move::bad();
    }

    from = square(algebra[0], algebra[1]);
    to = square(algebra[2], algebra[3]);
    
    if (algebra.size()!=4)
    {
        switch(algebra[4])
        {
        case 'n': promo=0x2; break;
        case 'b': promo=0x3; break;
        case 'r': promo=0x4; break;
        case 'q': promo=0x5; break;
        case 'N': promo=0x2; break;
        case 'B': promo=0x3; break;
        case 'R': promo=0x4; break;
        case 'Q': promo=0x5; break;
        default: throw move::bad();
        }
    }
}

move::move(
            int _from
        ,   int _to
        ,   int _promo
    )
        :   from(_from)
        ,   to(_to)
        ,   promo(_promo)
{}

bool move_t::operator<(const move_t& move) const
{
    return move.v() < v();
}

bool move_t::operator==(const move_t& move) const
{
    return m().to    == move.m().to
        && m().from  == move.m().from
        && m().promo == move.m().promo;
}

const move& move_t::m() const
{
    return *m_;
}

const float& move_t::v() const
{
    return *v_;
}

size_t logits::size() const
{
    return count_;
}

const move *logits::moves_data() const
{
    return moves_;
}

const float *logits::vals_data() const
{
    return vals_;
}

float logits::accuracy(const logits& logits, size_t K) const
{
    if (this->size()!=logits.size()) return 0.0f;
    if (this->size()<=K) return 1.0f;

    for (size_t i=0; i<K; i++)
    {
        if (pairs_[i]==logits.pairs_[i]) return 1.0f;
    }
    return 0.0f;
}

const move_t *logits::begin() const
{
    return pairs_;
}

const move_t *logits::end() const
{
    return pairs_+count_;
}

logits::logits()
    :   pairs_ {}
    ,   moves_ {}
    ,   vals_ {}
    ,   count_(0)
{}

logits::logits(const logits& Logits)
    :   logits(
                Logits.moves_
            ,   Logits.vals_
            ,   Logits.count_
        )
{}

logits::logits(
        const move moves[]
    ,   const float vals[]
    ,   size_t count
)
    :   pairs_ {}
    ,   moves_ {}
    ,   vals_ {}
    ,   count_(count)
{
    std::memcpy(moves_, moves, sizeof(move) * count);
    std::memcpy(vals_, vals, sizeof(float) * count);

    for (size_t i=0; i < count_; i++)
    {
        pairs_[i].m_=&moves_[i];
        pairs_[i].v_=&vals_[i];
    }

    std::sort(pairs_, pairs_+count_);
}

policy::policy(const std::string& pi)
    :   logits()
{
    std::string token;
    std::istringstream iss(pi);

    float sum=0.0f;

    while ((iss >> token))
    {
        moves_[count_]=move(token);

        if ((iss >> vals_[count_]))
        {
            sum += vals_[count_];
        }
        else {
            throw policy::bad();
        }
        count_++;
    }

    if (sum != 1) {
        throw policy::bad();
    }

    for (size_t i=0; i < count_; i++) 
    {
        pairs_[i].m_=&moves_[i];
        pairs_[i].v_=&vals_[i];
    }

    std::sort(pairs_, pairs_+count_);
}

policy::policy(const policy& pi)
    :   logits(pi)
{}