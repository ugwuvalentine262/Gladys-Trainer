/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef WDL_HPP
#define WDL_HPP

#include <string>

struct WDL
{
    const float win;
    const float draw;
    const float loss;

public:

    float q() const;

    WDL()=delete;
    WDL(const std::string& wdl);
    WDL(float w, float d, float l);
    WDL(const WDL& wdl);

};

#endif // WDL_HPP