#include "preemption_info.h"
#include "preempt_overhead.h"

preemption_info_t::preemption_info_t(unsigned sid)
  : m_sid(sid)
{
}

preemption_info_t::preemption_info_t(const shader_preempt_overhead& item)
  : m_sid(item.shader_id)
{
  for (std::map<unsigned, kernel_info_t::PreemptionTechnique>::const_iterator it = item.preemption_techniques.begin(), it_end = item.preemption_techniques.end();
       it != it_end; ++it) {
    switch (it->second) {
      case kernel_info_t::PREEMPTION_DRAIN:
        make_cta_drain(it->first);
        break;

      case kernel_info_t::PREEMPTION_SWITCH:
        make_cta_switch(it->first);
        break;

      case kernel_info_t::PREEMPTION_FLUSH:
        make_cta_flush(it->first);
        break;

      default:
        // unreachable
        assert(false);
    }
  }
}

