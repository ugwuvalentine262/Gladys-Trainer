#include <sstream>
#include <string>

#include "WDL.hpp"

static WDL parse(const std::string& wdl)
{
    std::istringstream iss(wdl);

    float w=0, d=0, l=0;

    iss >> w;
    iss >> d;
    iss >> l;

    return WDL(w, d, l);
}

float WDL::q() const
{
    return win - loss;
}

WDL::WDL(const std::string& wdl)
    :   WDL(parse(wdl))
{}

WDL::WDL(const WDL& wdl)
    :   win(wdl.win)
    ,   draw(wdl.draw)
    ,   loss(wdl.loss)   
{}

WDL::WDL(float w, float d, float l)
    :   win(w), draw(d), loss(l)
{}