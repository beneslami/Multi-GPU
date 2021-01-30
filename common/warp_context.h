#ifndef WARP_CONTEXT_H_
#define WARP_CONTEXT_H_

#include "types.h"
#include "warp_consts.h"

#include "shd_warp_t.h"

class warp_context_t {
public:
    warp_context_t(const shd_warp_t& warp, unsigned flat_cta_id, unsigned wid);

public:
    unsigned m_flat_cta_id;
    unsigned m_wid;
    unsigned m_dynamic_warp_id;

    address_type m_next_pc;
    unsigned n_completed;          // number of threads in warp completed
    std::bitset<MAX_WARP_SIZE> m_active_threads;

    unsigned m_n_atomic;           // number of outstanding atomic operations 
    bool     m_membar;             // if true, warp is waiting at memory barrier

    bool m_done_exit; // true once thread exit has been registered for threads in this warp
};

#endif // WARP_CONTEXT_H_

