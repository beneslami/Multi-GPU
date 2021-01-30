#include <cassert>

#include "even_scheduler.h"
#include "fixed_scheduler.h"
#include "smart_even_scheduler.h"

#include "../../common/kernel_info_t.h"

//#define _DEBUG_MK_SCHEDULER
#ifdef _DEBUG_MK_SCHEDULER
#define _DEBUG_REMOVING     (true)
#include <fstream>
static std::ofstream dbgout;
#endif // _DEBUG_MK_SCHEDULER

MKScheduler*
MKScheduler::Create(const SchedulerInfo& info)
{
#ifdef _DEBUG_MK_SCHEDULER
  dbgout.open("debug.mk");
#endif // _DEBUG_MK_SCHEDULER

  assert(info.has("SchedulerType"));
  std::vector<std::string> ST = info.get("SchedulerType");
  std::string& schedulerType = ST[0];
  if (schedulerType == "Even") {
    return new EvenScheduler();
  } else if (schedulerType == "Fixed") {
    std::list<unsigned> allocs;
    for (unsigned i = 1, i_end = ST.size(); i < i_end; ++i) {
      allocs.push_back(std::stoi(ST[i]));
    }
    return new FixedScheduler(allocs);
  } else if (schedulerType == "Smart") {
    assert(ST.size() == 3);
    return new SmartEvenScheduler(ST[1], ST[2]);
  } else {
    printf("Unknown SchedulerType: %s\n", schedulerType.c_str());
    assert(false);
  }

  // unreachable
  return NULL;
}

DynamicScheduler::DynamicScheduler(const std::string& _preempt, const std::string& _progress)
{
  if (_preempt == "Drain") {
    preempt_policy = PREEMPT_DRAIN_ONLY;
  } else if (_preempt == "Switch") {
    preempt_policy = PREEMPT_SWITCH_ONLY;
  } else if (_preempt == "Flush") {
    preempt_policy = PREEMPT_FLUSH_ONLY;
  } else if (_preempt == "DrainSwitch") {
    preempt_policy = PREEMPT_DRAIN_SWITCH;
  } else if (_preempt == "DrainFlush") {
    preempt_policy = PREEMPT_DRAIN_FLUSH;
  } else if (_preempt == "SwitchFlush") {
    preempt_policy = PREEMPT_SWITCH_FLUSH;
  } else if (_preempt == "DrainSwitchFlush") {
    preempt_policy = PREEMPT_DRAIN_SWITCH_FLUSH;
  } else if (_preempt == "FineGrain") {
    preempt_policy = PREEMPT_FINE_GRAIN;
  } else {
    printf("Unknown Preemption Policy: %s\n", _preempt.c_str());
    assert(false);
  }

  if (_progress == "Inst") {
    progress_policy = PROGRESS_INST;
  } else if (_progress == "Cycle") {
    progress_policy = PROGRESS_CYCLE;
  } else {
    printf("Unknown Progress Policy: %s\n", _progress.c_str());
    assert(false);
  }
}

kernel_info_t*
DynamicScheduler::next_thread_block_to_schedule()
{
  kernel_info_t* result = NULL;
  int max_diff = 0;

  for (std::map<kernel_info_t*, SM_info>::iterator it = SM_infos.begin(), it_end = SM_infos.end();
       it != it_end; ++it) {
    SM_info & info = it->second;
    int curr_diff = info.num_expected_shader() - info.num_holding_shader();
    if (curr_diff > max_diff && !it->first->no_more_ctas_to_run()) {
      result = it->first;
      max_diff = curr_diff;
    }
  }

  return result;
}

void
DynamicScheduler::inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_infos[kernel].add_shader(linear_shader_id);
}

void
DynamicScheduler::dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_infos[kernel].remove_shader(linear_shader_id);
}

void
DynamicScheduler::add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader)
{
  assert(SM_infos.find(kernel) == SM_infos.end());
  SM_infos[kernel].initialize(num_SMs);

  MKScheduler::add_kernel(kernel, max_cta_per_shader);
}

void
DynamicScheduler::remove_kernel(kernel_info_t* kernel)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_infos.erase(kernel);

  MKScheduler::remove_kernel(kernel);
}

std::vector<kernel_info_t*>
DynamicScheduler::check_for_removal() const
{
  std::vector<kernel_info_t*> kernels_with_remove;
  for (std::map<kernel_info_t*, SM_info>::const_iterator it = SM_infos.begin(), it_end = SM_infos.end();
       it != it_end; ++it) {
    const SM_info& info = it->second;
    // if holding more than the expected + removing
    // remove more
    if (info.num_holding_shader() > (info.num_expected_shader() + info.num_removing_shader()) && info.num_holding_shader() > 0) {
      kernels_with_remove.push_back(it->first);
    }
  }

  return kernels_with_remove;
}

std::vector<kernel_info_t*>
DynamicScheduler::check_for_cancel_removal() const
{
  std::vector<kernel_info_t*> kernels_with_cancel;
  for (std::map<kernel_info_t*, SM_info>::const_iterator it = SM_infos.begin(), it_end = SM_infos.end();
       it != it_end; ++it) {
    const SM_info& info = it->second;
    // if holding less than the expected + removing (removing more than necessary)
    // cancel removal
    if (info.num_holding_shader() < (info.num_expected_shader() + info.num_removing_shader()) && info.num_removing_shader() > 0) {
      kernels_with_cancel.push_back(it->first);
    }
  }

  return kernels_with_cancel;
}

unsigned
DynamicScheduler::get_num_remove_required(kernel_info_t* kernel) const
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  const SM_info& info = SM_infos.at(kernel);
  return std::min(info.num_holding_shader() - (info.num_expected_shader() + info.num_removing_shader()), info.num_holding_shader());
}

unsigned
DynamicScheduler::get_num_cancel_remove_required(kernel_info_t* kernel) const
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  const SM_info& info = SM_infos.at(kernel);
  return std::min((info.num_expected_shader() + info.num_removing_shader()) - info.num_holding_shader(), info.num_removing_shader());
}

std::vector<preemption_info_t>
DynamicScheduler::find_shaders_to_remove(kernel_info_t* kernel, unsigned num_remove)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  switch (preempt_policy) {
    case PREEMPT_DRAIN_ONLY: {
      std::vector<unsigned> most_progressed = kernel->find_most_progressed_shaders(info.get_drainable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<unsigned>::iterator it = most_progressed.begin(), it_end = most_progressed.end();
           it != it_end; ++it) {
        info.preempt_shader(*it);
        info.set_shader_to_drain(*it);

        result.push_back(preemption_info_t(*it));
        preemption_info_t& curr_preempt = result.back();
        for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
          if (info.is_cta_running(*it, i)) {
            curr_preempt.make_cta_drain(i);
          }
        }
      }

      return result;
    }

    case PREEMPT_SWITCH_ONLY: {
      std::vector<unsigned> least_progressed = kernel->find_most_progressed_shaders(info.get_switchable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<unsigned>::iterator it = least_progressed.begin(), it_end = least_progressed.end();
           it != it_end; ++it) {
        info.preempt_shader(*it);
        info.set_shader_to_switch(*it);

        result.push_back(preemption_info_t(*it));
        preemption_info_t& curr_preempt = result.back();
        for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
          if (info.is_cta_running(*it, i)) {
            curr_preempt.make_cta_switch(i);
          }
        }
      }
      return result;
    }

    case PREEMPT_FLUSH_ONLY: {
      std::vector<unsigned> least_progressed = kernel->find_least_progressed_shaders(info.get_flushable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<unsigned>::iterator it = least_progressed.begin(), it_end = least_progressed.end();
           it != it_end; ++it) {
        info.preempt_shader(*it);
        info.set_shader_to_flush(*it);

        result.push_back(preemption_info_t(*it));
        preemption_info_t& curr_preempt = result.back();
        for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
          if (info.is_cta_running(*it, i)) {
            curr_preempt.make_cta_flush(i);
          }
        }
      }
      return result;
    }

    case PREEMPT_DRAIN_SWITCH: {
      std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> > candidates = kernel->find_drain_switch_shaders(info.get_drainable_vector(), info.get_switchable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >::iterator it = candidates.begin(), it_end = candidates.end();
           it != it_end; ++it) {
        info.preempt_shader(it->first);
        result.push_back(preemption_info_t(it->first));
        preemption_info_t& curr_preempt = result.back();

        switch (it->second) {
        case kernel_info_t::PREEMPTION_DRAIN:
          info.set_shader_to_drain(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_drain(i);
            }
          }
          break;

        case kernel_info_t::PREEMPTION_SWITCH:
          info.set_shader_to_switch(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_switch(i);
            }
          }
          break;

        default:
          assert(false);
        }
      }
      return result;
    }

    case PREEMPT_DRAIN_FLUSH: {
      std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> > candidates = kernel->find_drain_flush_shaders(info.get_drainable_vector(), info.get_flushable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >::iterator it = candidates.begin(), it_end = candidates.end();
           it != it_end; ++it) {
        info.preempt_shader(it->first);
        result.push_back(preemption_info_t(it->first));
        preemption_info_t& curr_preempt = result.back();

        switch (it->second) {
        case kernel_info_t::PREEMPTION_DRAIN:
          info.set_shader_to_drain(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_drain(i);
            }
          }
          break;

        case kernel_info_t::PREEMPTION_FLUSH:
          info.set_shader_to_flush(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_flush(i);
            }
          }
          break;

        default:
          assert(false);
        }
      }
      return result;

    }

    case PREEMPT_SWITCH_FLUSH: {
      std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> > candidates = kernel->find_switch_flush_shaders(info.get_switchable_vector(), info.get_flushable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >::iterator it = candidates.begin(), it_end = candidates.end();
           it != it_end; ++it) {
        info.preempt_shader(it->first);
        result.push_back(preemption_info_t(it->first));
        preemption_info_t& curr_preempt = result.back();

        switch (it->second) {
        case kernel_info_t::PREEMPTION_SWITCH:
          info.set_shader_to_switch(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_switch(i);
            }
          }
          break;

        case kernel_info_t::PREEMPTION_FLUSH:
          info.set_shader_to_flush(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_flush(i);
            }
          }
          break;

        default:
          assert(false);
        }
      }
      return result;
    }

    case PREEMPT_DRAIN_SWITCH_FLUSH: {
      std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> > candidates = kernel->find_drain_switch_flush_shaders(info.get_drainable_vector(), info.get_switchable_vector(), info.get_flushable_vector(), num_remove);
      std::vector<preemption_info_t> result;
      for (std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >::iterator it = candidates.begin(), it_end = candidates.end();
           it != it_end; ++it) {
        info.preempt_shader(it->first);
        result.push_back(preemption_info_t(it->first));
        preemption_info_t& curr_preempt = result.back();

        switch (it->second) {
        case kernel_info_t::PREEMPTION_DRAIN:
          info.set_shader_to_drain(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_drain(i);
            }
          }
          break;

        case kernel_info_t::PREEMPTION_SWITCH:
          info.set_shader_to_switch(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_switch(i);
            }
          }
          break;

        case kernel_info_t::PREEMPTION_FLUSH:
          info.set_shader_to_flush(it->first);
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (info.is_cta_running(it->first, i)) {
              curr_preempt.make_cta_flush(i);
            }
          }
          break;

        default:
          assert(false);
        }
      }
      return result;
    }

    case PREEMPT_FINE_GRAIN: {
      std::vector<preemption_info_t> result = kernel->find_fine_grain_preemptions(info.get_drainable_vector(), info.get_non_idempotent_vector(), num_remove);
      for (std::vector<preemption_info_t>::iterator sit = result.begin(), sit_end = result.end();
           sit != sit_end; ++sit) {
        if (sit->get_preempting_ctas() > 0) {
          info.preempt_shader(sit->get_sid());
        }
        for (preemption_info_t::iterator tbit = sit->begin(), tbit_end = sit->end();
             tbit != tbit_end; ++tbit) {
          if (sit->is_cta_draining(tbit->first)) {
            info.set_shader_to_drain(sit->get_sid());
          } else if (sit->is_cta_switching(tbit->first)) {
            info.set_shader_to_switch(sit->get_sid());
          } else {
            assert(sit->is_cta_flushing(tbit->first));
            info.set_shader_to_flush(sit->get_sid());
          }
        }
      }
      return result;
    }

    default:
      assert(false);
  }

  // unreachable
  std::vector<preemption_info_t> no_candidates;
  return no_candidates;
}

std::vector<unsigned>
DynamicScheduler::find_shaders_to_cancel(kernel_info_t* kernel, unsigned num_cancel)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];

  std::vector<bool> candidates;
  candidates.resize(num_SMs, false);
  for (unsigned i = 0; i < num_SMs; ++i) {
    //if (info.is_draining(i) || info.is_flushing(i)) {
    // only cancel which is draining
    if (info.is_draining(i) && !info.is_flushing(i) && !info.is_context_saving(i)) {
      candidates[i] = true;
    }
  }

  switch (preempt_policy) {
    case PREEMPT_DRAIN_ONLY:
    case PREEMPT_DRAIN_SWITCH:
    case PREEMPT_DRAIN_FLUSH:
    case PREEMPT_DRAIN_SWITCH_FLUSH:
    {
      // right now, we cancel only drain!
      std::vector<unsigned> least_progressed = kernel->find_least_progressed_shaders(candidates, num_cancel);
      for (std::vector<unsigned>::iterator it = least_progressed.begin(), it_end = least_progressed.end();
           it != it_end; ++it) {
        info.cancel_removal(*it);
      }
      return least_progressed;
    }

    case PREEMPT_SWITCH_ONLY:
    case PREEMPT_FLUSH_ONLY:
    case PREEMPT_SWITCH_FLUSH:
    case PREEMPT_FINE_GRAIN:
      // Do not cancel
      break;

    default:
      assert(false);
  }

  std::vector<unsigned> no_candidates;
  return no_candidates;
}

void
DynamicScheduler::shader_loads_context(kernel_info_t* kernel, unsigned shader_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.make_context_load(shader_id);
}

void
DynamicScheduler::shader_finishes_loading_context(kernel_info_t* kernel, unsigned shader_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.make_context_load_done(shader_id);
}

void
DynamicScheduler::start_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.cta_starts_running(shader_id, cta_id);
}

void
DynamicScheduler::stop_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.cta_stops_running(shader_id, cta_id);
}

void
DynamicScheduler::finish_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.cta_finishes_running(shader_id, cta_id);
}

void
DynamicScheduler::CTA_executes_atomic(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.cta_passes_non_idempotent_region(shader_id, cta_id);
}

void
DynamicScheduler::CTA_overwrites_input(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)
{
  assert(SM_infos.find(kernel) != SM_infos.end());
  SM_info& info = SM_infos[kernel];
  info.cta_passes_non_idempotent_region(shader_id, cta_id);
}

void
DynamicScheduler::SM_info::add_shader(unsigned linear_shader_id)
{
  assert(!occupied[linear_shader_id]);
  occupied[linear_shader_id] = true;
  ++holding_shaders;
}

void
DynamicScheduler::SM_info::remove_shader(unsigned linear_shader_id)
{
  assert(occupied[linear_shader_id] || is_draining(linear_shader_id) || is_context_saving(linear_shader_id) || is_flushing(linear_shader_id));
  occupied[linear_shader_id] = false;
  --holding_shaders;

  if (to_remove[linear_shader_id].is_being_removed) {
    to_remove[linear_shader_id].reset();
    --removing_shaders;

#ifdef _DEBUG_MK_SCHEDULER
    if (_DEBUG_REMOVING) {
      dbgout << "Shader[" << linear_shader_id << "], is removed!" << std::endl;
      printf("GPGPU-Sim MK: Shader[%d], is removed!\n" , linear_shader_id);
    }
#endif // _DEBUG_MK_SCHEDULER
  }
}

void
DynamicScheduler::SM_info::preempt_shader(unsigned linear_shader_id)
{
  assert(!to_remove[linear_shader_id].is_being_removed);
  occupied[linear_shader_id] = false;
  ++removing_shaders;
}

void
DynamicScheduler::SM_info::set_shader_to_drain(const unsigned linear_shader_id)
{
  to_remove[linear_shader_id].make_drain();

#ifdef _DEBUG_MK_SCHEDULER
  if (_DEBUG_REMOVING) {
    dbgout << "Shader[" << linear_shader_id << "] will be drained!" << std::endl;
    printf("GPGPU-Sim MK: Shader[%d] will be drained!\n" , linear_shader_id);
  }
#endif // _DEBUG_MK_SCHEDULER
}

void
DynamicScheduler::SM_info::set_shader_to_switch(const unsigned linear_shader_id)
{
  to_remove[linear_shader_id].make_switch();

#ifdef _DEBUG_MK_SCHEDULER
  if (_DEBUG_REMOVING) {
    dbgout << "Shader[" << linear_shader_id << "] will be switched!" << std::endl;
    printf("GPGPU-Sim MK: Shader[%d] will be switched!\n" , linear_shader_id);
  }
#endif // _DEBUG_MK_SCHEDULER
}

void
DynamicScheduler::SM_info::set_shader_to_flush(const unsigned linear_shader_id)
{
  to_remove[linear_shader_id].make_flush();

#ifdef _DEBUG_MK_SCHEDULER
  if (_DEBUG_REMOVING) {
    dbgout << "Shader[" << linear_shader_id << "] will be flushed!" << std::endl;
    printf("GPGPU-Sim MK: Shader[%d] will be flushed!\n" , linear_shader_id);
  }
#endif // _DEBUG_MK_SCHEDULER
}

void
DynamicScheduler::SM_info::set_upper_bound(const unsigned bound)
{
  expected_shaders = std::min(bound, maximum_alloc);
}

void
DynamicScheduler::SM_info::set_initial_alloc(const unsigned alloc)
{
  assert(alloc <= maximum_alloc);
  maximum_alloc = alloc;
  expected_shaders = alloc;
}

void
DynamicScheduler::SM_info::cancel_removal(const unsigned linear_shader_id)
{
#ifdef _DEBUG_MK_SCHEDULER
  if (_DEBUG_REMOVING) {
    dbgout << "Shader[" << linear_shader_id << "], released from removal!" << std::endl;
    printf("GPGPU-Sim MK: Shader[%d], released from removal!\n" , linear_shader_id);
  }
#endif // _DEBUG_MK_SCHEDULER

  assert(to_remove[linear_shader_id].is_being_removed);
  assert(removing_shaders > 0);
  to_remove[linear_shader_id].reset();
  occupied[linear_shader_id] = true;
  --removing_shaders;
}

void
DynamicScheduler::SM_info::cta_starts_running(const unsigned shader_id, const unsigned cta_id)
{
  assert(!cta_running[shader_id][cta_id]);
  assert(!cta_non_idempotent[shader_id][cta_id]);
  cta_running[shader_id].set(cta_id);
}

void
DynamicScheduler::SM_info::cta_stops_running(const unsigned shader_id, const unsigned cta_id)
{
  assert(cta_running[shader_id][cta_id]);
  cta_running[shader_id].reset(cta_id);
  cta_non_idempotent[shader_id].reset(cta_id);
}

void
DynamicScheduler::SM_info::cta_finishes_running(const unsigned shader_id, const unsigned cta_id)
{
  assert(cta_running[shader_id][cta_id]);
  cta_running[shader_id].reset(cta_id);
  cta_non_idempotent[shader_id].reset(cta_id);
}

void
DynamicScheduler::SM_info::cta_passes_non_idempotent_region(const unsigned shader_id, const unsigned cta_id)
{
  assert(cta_running[shader_id][cta_id]);
  cta_non_idempotent[shader_id].set(cta_id);
}

bool
DynamicScheduler::SM_info::is_context_loading(const unsigned linear_shader_id) const
{
  assert(!is_to_be_removed(linear_shader_id));
  return context_loading[linear_shader_id];
}

void
DynamicScheduler::SM_info::make_context_load(const unsigned linear_shader_id)
{
  assert(!is_to_be_removed(linear_shader_id) && !is_context_loading(linear_shader_id));
  context_loading[linear_shader_id] = true;
}

void
DynamicScheduler::SM_info::make_context_load_done(const unsigned linear_shader_id)
{
  assert(!is_to_be_removed(linear_shader_id) && is_context_loading(linear_shader_id));
  context_loading[linear_shader_id] = false;
}

