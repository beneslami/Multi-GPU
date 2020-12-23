#include "kernel_info_t.h"
#include <algorithm>
#include <set>
#include "preempt_overhead.h"

//#define _DEBUG_KERNEL_INFO
#ifdef _DEBUG_KERNEL_INFO
#define _DEBUG_PAUSE          (false)
#define _DEBUG_CHIMERA        (true)
#include <fstream>
static std::ofstream dbgout;
static void open_dbgfile()
{
  if (!dbgout.is_open()) {
    dbgout.open("debug.kernel_info");
  }
}

#endif // _DEBUG_KERNEL_INFO

extern unsigned long long get_curr_cycle();
unsigned kernel_info_t::m_next_uid = 1;

// prev_kernel_stats[ChildProcess*][name] = average cta cycles
static std::map<ChildProcess*, std::map<std::string, double> > prev_kernel_stats;

static void
increment_x_then_y_then_z( dim3 &i, const dim3 &bound)
{
   i.x++;
   if ( i.x >= bound.x ) {
      i.x = 0;
      i.y++;
      if ( i.y >= bound.y ) {
         i.y = 0;
         if( i.z < bound.z ) 
            i.z++;
      }
   }
}

void
kernel_info_t::increment_cta_id(unsigned core_id) 
{
  // First, check if any context switching CTA
  // Second, check if any context flushed CTA
  // Third, proceed normally
  if (!m_to_be_switched.empty() && (m_to_be_switched.top().ready_time <= get_curr_cycle())) {
    m_to_be_switched.pop();
  } else if (!m_to_be_migrated.empty()) {
    m_to_be_migrated.pop_back();
  } else {
//    increment_x_then_y_then_z(m_next_cta,m_grid_dim);
    if(kain_m_next_cta[core_id/32].size() != 0)
        kain_m_next_cta[core_id/32].pop_front();
  }
  m_next_tid.x=0;
  m_next_tid.y=0;
  m_next_tid.z=0;
}

void
kernel_info_t::increment_thread_id()
{
  increment_x_then_y_then_z(m_next_tid,m_block_dim);
}

size_t
kernel_info_t::executed_blocks() const
{
    unsigned remained_blocks = 0;
    for(int i = 0; i < 4; i++)
    {   
       remained_blocks += kain_m_next_cta[i].size(); 
    }   
  return   (num_blocks()-remained_blocks) - m_to_be_migrated.size() - m_to_be_switched.size();
}

void
kernel_info_t::check_for_prev_stats()
{
  assert(m_parent != NULL);
  std::string kernel_name = name();
  if (prev_kernel_stats[m_parent].find(kernel_name) != prev_kernel_stats[m_parent].end()) {
    m_cta_avg_cycle = prev_kernel_stats[m_parent][kernel_name];
    assert(m_finished_ctas == 0);
    m_finished_ctas = 1;

    printf("GPGPU-Sim Kernel: Process [0x%p] relaunching %s with avg_cycle = %f!\n", m_parent, kernel_name.c_str(), m_cta_avg_cycle);
  } else {
    printf("GPGPU-Sim Kernel: Process [0x%p] launches %s for the first time!\n", m_parent, kernel_name.c_str());
  }
}

long long kain_warp_inst_app1 = 0;
long long kain_warp_inst_app2 = 0;
void
kernel_info_t::inc_num_simulated_insts(unsigned active_count)
{
  m_num_simulated_insts += active_count;

  assert(m_parent != NULL);
  // ChildProcess is aware whether it was finished or not
  m_parent->inc_num_simulated_insts(active_count);


  if(get_kain_stream_id() == 1)
  		kain_warp_inst_app1 += 1;
  if(get_kain_stream_id() == 2)
  		kain_warp_inst_app2 += 1;

}

void
kernel_info_t::dec_num_simulated_insts(unsigned wasted_insts)
{
  m_num_simulated_insts -= wasted_insts;

  assert(m_parent != NULL);
  // ChildProcess is aware whether it was finished or not
  m_parent->dec_num_simulated_insts(wasted_insts);
}

void
kernel_info_t::set_init_max_cta_per_shader(unsigned _max_cta,unsigned core_sid)
{
  init_max_cta_per_shader = _max_cta; 
  //assert(core_sid < cta_per_shader.size());
//  for (unsigned i = 0, i_end = cta_per_shader.size(); i < i_end; ++i) {
    cta_per_shader[core_sid] = _max_cta;
 // }
}

void
kernel_info_t::set_max_cta_per_shader(const unsigned shader_id, const unsigned _max_cta)
{
  assert(_max_cta > 0 && _max_cta <= init_max_cta_per_shader);
  for (unsigned i = 0, i_end = cta_per_shader.size(); i < i_end; ++i) {
    if (_max_cta < cta_per_shader[i]) {
      cta_per_shader[i] = _max_cta;
    }
  }

  cta_per_shader[shader_id] = _max_cta;
}

unsigned
kernel_info_t::get_required_shaders() const
{
  size_t remaining_blocks = num_blocks() + m_running_ctas - executed_blocks();
  unsigned max_cta_per_shader = init_max_cta_per_shader;
  for (unsigned i = 0, i_end = cta_per_shader.size(); i < i_end; ++i) {
    if (cta_per_shader[i] < max_cta_per_shader) {
      max_cta_per_shader = cta_per_shader[i];
    }
  }
  unsigned required_shaders = remaining_blocks / max_cta_per_shader;
  if (remaining_blocks > (required_shaders * max_cta_per_shader)) {
    // there is remainder
    ++required_shaders;
  }

  return required_shaders;
}

void
kernel_info_t::initialize_with_num_shaders(const unsigned num_shader, const unsigned num_cta_per_shader)
{
  cta_per_shader.resize(num_shader, 0);
  scheduled_cta_per_shader.resize(num_shader, 0);
  m_cta_statistics.resize(num_shader);
  for (unsigned i = 0; i < num_shader; ++i) {
    m_cta_statistics[i].resize(num_cta_per_shader);
  }
}

void
kernel_info_t::inc_scheduled_cta(const unsigned shader_id)
{
  assert(scheduled_cta_per_shader[shader_id] < cta_per_shader[shader_id]);
  ++scheduled_cta_per_shader[shader_id];
}

void
kernel_info_t::dec_scheduled_cta(const unsigned shader_id)
{
  assert(scheduled_cta_per_shader[shader_id] > 0);
  --scheduled_cta_per_shader[shader_id];
}

std::vector<unsigned>
kernel_info_t::find_least_progressed_shaders(const std::vector<bool>& candidates, unsigned num_shaders) const
{
  assert(candidates.size() == m_cta_statistics.size());
  unsigned num_candidates = 0;
  for (unsigned i = 0; i < candidates.size(); ++i) {
    if (candidates[i]) {
      ++num_candidates;
    }
  }

  const unsigned num_to_pick = std::min(num_candidates, num_shaders);
  std::vector<unsigned> result_shaders;
  for (unsigned i = 0; i < num_to_pick; ++i) {
    unsigned long long least_progress = (unsigned long long)-1;
    unsigned least_progress_shader = (unsigned)-1;
    for (unsigned shader_id = 0; shader_id < m_cta_statistics.size(); ++shader_id) {
      if (candidates[shader_id] && std::find(result_shaders.begin(), result_shaders.end(), shader_id) == result_shaders.end()) {
        unsigned long long curr_exec_insts = get_executed_insts_for_shader(shader_id);
        if (curr_exec_insts < least_progress) {
          least_progress = curr_exec_insts;
          least_progress_shader = shader_id;
        }
      }
    }

    assert(least_progress_shader != (unsigned)-1);
    result_shaders.push_back(least_progress_shader);
  }
  return result_shaders;
}

std::vector<unsigned>
kernel_info_t::find_most_progressed_shaders(const std::vector<bool>& candidates, unsigned num_shaders) const
{
  assert(candidates.size() == m_cta_statistics.size());
  unsigned num_candidates = 0;
  for (unsigned i = 0; i < candidates.size(); ++i) {
    if (candidates[i]) {
#ifdef _DEBUG_KERNEL_INFO
      if (_DEBUG_CHIMERA) {
        open_dbgfile();
        dbgout << "@" << get_curr_cycle() << ", Shader[" << i << "] can be drained: lat = " << get_max_remaining_cycle_for_shader(i) << ", tp = " << get_draining_throughput_overhead_for_shader(i) << std::endl;
      }
#endif // _DEBUG_KERNEL_INFO

      ++num_candidates;
    }
  }

  const unsigned num_to_pick = std::min(num_candidates, num_shaders);
  std::vector<unsigned> result_shaders;
  for (unsigned i = 0; i < num_to_pick; ++i) {
    unsigned long long most_progress = 0;
    unsigned most_progress_shader = (unsigned)-1;
    for (unsigned shader_id = 0; shader_id < m_cta_statistics.size(); ++shader_id) {
      if (candidates[shader_id] && std::find(result_shaders.begin(), result_shaders.end(), shader_id) == result_shaders.end()) {
        unsigned long long curr_exec_insts = get_executed_insts_for_shader(shader_id);
        if (curr_exec_insts >= most_progress) {
          most_progress = curr_exec_insts;
          most_progress_shader = shader_id;
        }
      }
    }

    assert(most_progress_shader != (unsigned)-1);
    result_shaders.push_back(most_progress_shader);
  }

  return result_shaders;
}

void
kernel_info_t::start_cta(unsigned shader_id, unsigned hw_cta_id)
{
  //assert(shader_id < m_cta_statistics.size());
  assert(hw_cta_id < m_cta_statistics[shader_id].size());
  assert(!m_cta_statistics[shader_id][hw_cta_id].is_valid());
  m_cta_statistics[shader_id][hw_cta_id].start_stat();
}

void
kernel_info_t::restart_cta(unsigned shader_id, unsigned hw_cta_id, const cta_stat_context_t& stored_stat)
{
  //assert(shader_id < m_cta_statistics.size());
  assert(hw_cta_id < m_cta_statistics[shader_id].size());
  assert(!m_cta_statistics[shader_id][hw_cta_id].is_valid());
  m_cta_statistics[shader_id][hw_cta_id].load_context(stored_stat);
}

void
kernel_info_t::finish_cta(unsigned shader_id, unsigned hw_cta_id)
{
  //assert(shader_id < m_cta_statistics.size());
  assert(hw_cta_id < m_cta_statistics[shader_id].size());
  assert(m_cta_statistics[shader_id][hw_cta_id].is_valid());
  // deal with insts first
  register_cta_num_insts(m_cta_statistics[shader_id][hw_cta_id].get_executed_insts());
  // deal with cycles
  unsigned long long curr_executed_cycles = m_cta_statistics[shader_id][hw_cta_id].get_executed_cycles();
  m_cta_avg_cycle = (m_cta_avg_cycle * double(m_finished_ctas) + double(curr_executed_cycles)) / double(m_finished_ctas + 1);
  prev_kernel_stats[get_parent_process()][name()] = m_cta_avg_cycle;
  ++m_finished_ctas;
  // clear stat
  m_cta_statistics[shader_id][hw_cta_id].clear_context();
}

void
kernel_info_t::stop_cta(unsigned shader_id, unsigned hw_cta_id)
{
  //assert(shader_id < m_cta_statistics.size());
  assert(hw_cta_id < m_cta_statistics[shader_id].size());
  assert(m_cta_statistics[shader_id][hw_cta_id].is_valid());
  m_cta_statistics[shader_id][hw_cta_id].clear_context();
}

void
kernel_info_t::inc_cta_inst(unsigned shader_id, unsigned hw_cta_id)
{
  //assert(shader_id < m_cta_statistics.size());
  assert(hw_cta_id < m_cta_statistics[shader_id].size());
  assert(m_cta_statistics[shader_id][hw_cta_id].is_valid());
  m_cta_statistics[shader_id][hw_cta_id].inc_executed_insts();
}

void
kernel_info_t::context_switch_cta(dim3 switching_cta_id, unsigned long long context_save_done_time, unsigned shader_id, unsigned hw_cta_id)
{
  m_to_be_switched.push(timed_dim3(switching_cta_id, context_save_done_time));
}

static bool
dim3_equals(const dim3& lhs, const dim3& rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool
kernel_info_t::is_context_switching_cta(const dim3& cta_id) const
{
  return !m_to_be_switched.empty() && dim3_equals(m_to_be_switched.top().cta_id, cta_id);
}

dim3
kernel_info_t::get_next_cta_id(unsigned core_id) const
{
  if (!m_to_be_switched.empty() && (m_to_be_switched.top().ready_time <= get_curr_cycle())) {
    return m_to_be_switched.top().cta_id;
  } else if (!m_to_be_migrated.empty()) {
    return m_to_be_migrated.back();
  } else {
    //assert(core_id < 128);
    if(kain_m_next_cta[core_id/32].size() != 0)
        return kain_m_next_cta[core_id/32].front();
  }
}

bool
kernel_info_t::can_issue_new_cta(unsigned core_id) const
{
    bool can_issue = false;

    if(kain_m_next_cta[core_id/32].size() != 0)
         can_issue = true;

  return can_issue || !m_to_be_migrated.empty() || (!m_to_be_switched.empty() && (m_to_be_switched.top().ready_time <= get_curr_cycle()));
}

std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >
kernel_info_t::find_drain_switch_shaders(const std::vector<bool>& drain_candidates, const std::vector<bool>& switch_candidates, unsigned num_shaders) const
{
  // we are dealing with equal number of SMs
  assert(drain_candidates.size() == m_cta_statistics.size());

  // add to sorted queue
  sort_throughput_queue sorted_candidates;
  for (unsigned i = 0; i < drain_candidates.size(); ++i) {
    if (drain_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_draining_throughput_overhead_for_shader(i), get_max_remaining_cycle_for_shader(i), PREEMPTION_DRAIN));
    }
  }

  for (unsigned i = 0; i < switch_candidates.size(); ++i) {
    if (switch_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_switching_throughput_overhead_for_shader(i), get_switching_overhead_latency_for_shader(i), PREEMPTION_SWITCH));
    }
  }

  // generate throughput related result
  return select_candidates(num_shaders, sorted_candidates);
}

std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >
kernel_info_t::find_drain_flush_shaders(const std::vector<bool>& drain_candidates, const std::vector<bool>& flush_candidates, unsigned num_shaders) const
{
  // we are dealing with equal number of SMs
  assert(drain_candidates.size() == m_cta_statistics.size());

  // add to sorted queue
  sort_throughput_queue sorted_candidates;
  for (unsigned i = 0; i < drain_candidates.size(); ++i) {
    if (drain_candidates[i]) {
#ifdef _DEBUG_KERNEL_INFO
      if (_DEBUG_CHIMERA) {
        open_dbgfile();
        dbgout << "@" << get_curr_cycle() << ", Shader[" << i << "] can be drained: lat = " << get_max_remaining_cycle_for_shader(i) << ", tp = " << get_draining_throughput_overhead_for_shader(i) << std::endl;
      }
#endif // _DEBUG_KERNEL_INFO
      sorted_candidates.push(preempt_overhead_item(i, get_draining_throughput_overhead_for_shader(i), get_max_remaining_cycle_for_shader(i), PREEMPTION_DRAIN));
    }
  }

  for (unsigned i = 0; i < flush_candidates.size(); ++i) {
    if (flush_candidates[i]) {
#ifdef _DEBUG_KERNEL_INFO
      if (_DEBUG_CHIMERA) {
        open_dbgfile();
        dbgout << "@" << get_curr_cycle() << ", Shader[" << i << "] can be flushed: lat = " << 0 << ", tp = " << get_executed_insts_for_shader(i) << std::endl;
      }
#endif // _DEBUG_KERNEL_INFO
      sorted_candidates.push(preempt_overhead_item(i, get_executed_insts_for_shader(i), 0, PREEMPTION_FLUSH));
    }
  }

  // generate result
  return select_candidates(num_shaders, sorted_candidates);
}

std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >
kernel_info_t::find_switch_flush_shaders(const std::vector<bool>& switch_candidates, const std::vector<bool>& flush_candidates, unsigned num_shaders) const
{
  // we are dealing with equal number of SMs
  assert(switch_candidates.size() == m_cta_statistics.size());

  // add to sorted queue
  sort_throughput_queue sorted_candidates;
  for (unsigned i = 0; i < switch_candidates.size(); ++i) {
    if (switch_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_switching_throughput_overhead_for_shader(i), get_switching_overhead_latency_for_shader(i), PREEMPTION_SWITCH));
    }
  }

  for (unsigned i = 0; i < flush_candidates.size(); ++i) {
    if (flush_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_executed_insts_for_shader(i), 0, PREEMPTION_FLUSH));
    }
  }

  // generate result
  return select_candidates(num_shaders, sorted_candidates);
}

std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >
kernel_info_t::find_drain_switch_flush_shaders(const std::vector<bool>& drain_candidates, const std::vector<bool>& switch_candidates, const std::vector<bool>& flush_candidates, unsigned num_shaders) const
{
  // we are dealing with equal number of SMs
  assert(drain_candidates.size() == m_cta_statistics.size());

  // add to sorted queue
  sort_throughput_queue sorted_candidates;
  for (unsigned i = 0; i < drain_candidates.size(); ++i) {
    if (drain_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_draining_throughput_overhead_for_shader(i), get_max_remaining_cycle_for_shader(i), PREEMPTION_DRAIN));
    }
  }

  for (unsigned i = 0; i < switch_candidates.size(); ++i) {
    if (switch_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_switching_throughput_overhead_for_shader(i), get_switching_overhead_latency_for_shader(i), PREEMPTION_SWITCH));
    }
  }

  for (unsigned i = 0; i < flush_candidates.size(); ++i) {
    if (flush_candidates[i]) {
      sorted_candidates.push(preempt_overhead_item(i, get_executed_insts_for_shader(i), 0, PREEMPTION_FLUSH));
    }
  }

  // generate result
  return select_candidates(num_shaders, sorted_candidates);
}

std::vector<preemption_info_t>
kernel_info_t::find_fine_grain_preemptions(const std::vector<bool>& candidates, const std::vector<std::bitset<MAX_CTA_PER_SHADER> >& non_idempotent_ctas, unsigned num_shaders) const
{
  shader_sort shader_overheads;
  for (unsigned sid = 0; sid < candidates.size(); ++sid) {
      if (candidates[sid]) {
          const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[sid];
          cta_sort cta_overheads;
          for (unsigned tbid = 0; tbid < MAX_CTA_PER_SHADER; ++tbid) {
              if (shader_stat[tbid].is_valid()) {
                  if (can_estimate_drain()) {
                    cta_overheads.push(cta_preempt_overhead(tbid, get_draining_throughput_overhead(sid, tbid), get_draining_latency(sid, tbid), PREEMPTION_DRAIN));
                  }
                  cta_overheads.push(cta_preempt_overhead(tbid, switching_overhead_per_cta, get_switching_throughput_overhead(sid, tbid), PREEMPTION_SWITCH));
                  if (!non_idempotent_ctas[sid][tbid]) {
                      cta_overheads.push(cta_preempt_overhead(tbid, get_executed_insts(sid, tbid), 0, PREEMPTION_FLUSH));
                  }
              }
          }

          shader_preempt_overhead curr_shader_overhead(sid);
          while (!cta_overheads.empty()) {
            const cta_preempt_overhead& item = cta_overheads.top();
            if (!curr_shader_overhead.cta_exists(item.hw_cta_id) && curr_shader_overhead.meets_latency_constraint(item)) {
              curr_shader_overhead.add_cta_with_preemption(item);
            }
            cta_overheads.pop();
          }

          // if non-selected ctas exist, context switch
          const unsigned ctas_to_preempt = num_valid_ctas(sid);
          if (curr_shader_overhead.size() < ctas_to_preempt) {
              for (unsigned tbid = 0; tbid < MAX_CTA_PER_SHADER; ++tbid) {
                  if (shader_stat[tbid].is_valid() && !curr_shader_overhead.cta_exists(tbid)) {
                      curr_shader_overhead.add_cta_with_preemption(cta_preempt_overhead(tbid, switching_overhead_per_cta, get_switching_throughput_overhead(sid, tbid), PREEMPTION_SWITCH));
                  }
              }
          }

          shader_overheads.push(curr_shader_overhead);
      }
  }

  return select_candidates(num_shaders, shader_overheads);
}

unsigned long long
kernel_info_t::get_max_remaining_cycle_for_shader(unsigned shader_id) const
{
#ifdef _DEBUG_KERNEL_INFO
  if (_DEBUG_CHIMERA) {
    open_dbgfile();
    dbgout << "  Shader[" << shader_id << "]: avg = " << m_cta_avg_cycle << " cycles, count = " << m_finished_ctas << std::endl;
  }
#endif // _DEBUG_KERNEL_INFO

  if (m_finished_ctas == 0) {
    // if we do not know how much longer it will take, draining will not be considered
    return (unsigned long long)-1;
  }

  //assert(shader_id < m_cta_statistics.size());
  const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[shader_id];

  unsigned long long max_start_cycle = 0;
  for (unsigned i = 0; i < shader_stat.size(); ++i) {
    if (shader_stat[i].is_valid()) {
      max_start_cycle = std::max(max_start_cycle, shader_stat[i].get_started_cycle());
    }
  }

  const unsigned long long cta_exec_cycle = (unsigned long long)m_cta_avg_cycle;
  const unsigned long long min_progressed_cycle = get_curr_cycle() - max_start_cycle;
  if (cta_exec_cycle > min_progressed_cycle) {

#ifdef _DEBUG_KERNEL_INFO
    if (_DEBUG_CHIMERA) {
      open_dbgfile();
      dbgout << "  Shader[" << shader_id << "]: max_start = " << max_start_cycle << ", min_progress = " << cta_exec_cycle - min_progressed_cycle << std::endl;
    }
#endif // _DEBUG_KERNEL_INFO

    return cta_exec_cycle - min_progressed_cycle;
  }

#ifdef _DEBUG_KERNEL_INFO
    if (_DEBUG_CHIMERA) {
      open_dbgfile();
      dbgout << "  Shader[" << shader_id << "]: max_start = " << max_start_cycle << ", min_progress = 0" << std::endl;
    }
#endif // _DEBUG_KERNEL_INFO

  // if progressed more than average, it is probably almost done
  // return 0 to get it drained
  return 0;
}

unsigned long long
kernel_info_t::get_switching_overhead_latency_for_shader(unsigned shader_id) const
{
  //assert(shader_id < m_cta_statistics.size());
  return switching_overhead_per_cta * num_valid_ctas(shader_id);
}

unsigned long long
kernel_info_t::get_draining_throughput_overhead_for_shader(unsigned shader_id) const
{
  if (m_finished_ctas == 0) {
    // if we do not know how much longer it will take, draining will not be considered
    return (unsigned long long)-1;
  }

  //assert(shader_id < m_cta_statistics.size());
  const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[shader_id];

  unsigned long long min_exec_insts = (unsigned long long)-1;
  for (unsigned i = 0; i < shader_stat.size(); ++i) {
    if (shader_stat[i].is_valid()) {
      min_exec_insts = std::min(min_exec_insts, shader_stat[i].get_executed_insts());
    }
  }

  unsigned long long overhead = 0;
  for (unsigned i = 0; i < shader_stat.size(); ++i) {
    if (shader_stat[i].is_valid()) {
      overhead += shader_stat[i].get_executed_insts() - min_exec_insts;
    }
  }

  return overhead;
}

unsigned long long
kernel_info_t::get_switching_throughput_overhead_for_shader(unsigned shader_id) const
{
  unsigned long long cycles = 2 * switching_overhead_per_cta * num_valid_ctas(shader_id);
  double ipc = 0.0;
  if (m_cta_num_insts == 0) {
    ipc = double(get_executed_insts_for_shader(shader_id)) / double(get_executed_cycles_for_shader(shader_id));
  } else {
    ipc = double(m_cta_num_insts) / m_cta_avg_cycle;
  }
  unsigned long long overhead = (unsigned long long)(ipc * double(cycles));
  return overhead;
}

