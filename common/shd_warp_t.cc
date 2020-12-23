#include "shd_warp_t.h"

bool
shd_warp_t::functional_done() const
{
    return get_n_completed() == m_warp_size;
}

bool
shd_warp_t::hardware_done() const
{
    return functional_done() && stores_done() && !inst_in_pipeline();
}

