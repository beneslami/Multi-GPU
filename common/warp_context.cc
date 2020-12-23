#include "warp_context.h"

warp_context_t::warp_context_t(const shd_warp_t& warp, unsigned flat_cta_id, unsigned wid)
{
  m_flat_cta_id = flat_cta_id;
  m_wid = wid;
  m_dynamic_warp_id = warp.get_dynamic_warp_id();

  m_next_pc = warp.get_pc();
  n_completed = warp.get_n_completed();
  m_active_threads = warp.get_active_threads();

  m_n_atomic = warp.get_n_atomic();
  m_membar = warp.get_membar();

  m_done_exit = warp.done_exit();

  assert(m_n_atomic == 0);
  assert(!m_membar);
}

