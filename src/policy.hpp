/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef POLICY_HPP
#define POLICY_HPP

#include <string>

struct move
{
    struct bad {};

	int from=-1;
	int to=-1;
	int promo=-1;

public:

    move()=default;
    move(const std::string& algebra, bool white);
    move(
            int _from
        ,   int _to
        ,   int _promo
    );

};

class move_t
{
    friend class logits;
    friend class policy;

    const move *m_;
    const float *v_;

public:

    bool operator<(const move_t&) const;
    bool operator==(const move_t&) const;
    const move& m() const;
    const float& v() const;

};

class logits
{

protected:

    move_t pairs_[128];
    move moves_[128];
    alignas(16) float vals_[128];
    size_t count_;

public:

    size_t size() const;

    const move *moves_data() const;
    const float *vals_data() const;

    float accuracy(const logits& logits, size_t K) const;

    const move_t *begin() const;
    const move_t *end() const;

    logits();
    logits(const logits& logits);
    logits(
            const move[]
        ,   const float[]
        ,   size_t count
    );
};

struct policy : public logits
{
    struct bad {};

    policy()=default;
    policy(const std::string& pi, bool white);
    policy(const policy& pi);

};

#endif // POLICY_HPP