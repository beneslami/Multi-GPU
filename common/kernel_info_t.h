#ifndef KERNEL_INFO_T_H
#define KERNEL_INFO_T_H

#include <list>
#include <queue>
#include <string>
#include <bitset>
#include <assert.h>

#include "../launcher/child_process.h"
#include "cta_stat_context.h"
#include "hard_consts.h"
#include "preemption_info.h"

#if !defined(__VECTOR_TYPES_H__) && !defined(__DIM3_DEFINED__)
struct dim3 {
   unsigned int x, y, z;
};
#define __DIM3_DEFINED__
#endif

class timed_dim3 {
public:
  timed_dim3(const dim3& _cta_id, unsigned long long _ready_time)
    : cta_id(_cta_id), ready_time(_ready_time)
  {}

  dim3 cta_id;
  unsigned long long ready_time;
};

class timed_dim3_comparison {
public:
  timed_dim3_comparison() {}
  bool operator() (const timed_dim3& lhs, const timed_dim3& rhs) const
  {
    return lhs.ready_time > rhs.ready_time;
  }
};

class kernel_info_t {
public:
   kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry, bool real_run = true );
   ~kernel_info_t();

   void inc_running(unsigned shader_id) { m_num_cores_running++; }
   void dec_running()
   {
       assert( m_num_cores_running > 0 );
       m_num_cores_running--;
   }
   bool running() const { return m_num_cores_running > 0; }
   bool done() const
   {
       bool kain_done = true;
       for(int i = 0; i < 4; i++)
            if(kain_m_next_cta[i].size() != 0)
                kain_done = false;

       return kain_done && !running();
   }
   class function_info *entry() { return m_kernel_entry; }
   const class function_info *entry() const { return m_kernel_entry; }

   size_t num_blocks() const
   {
      return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
   }

   size_t threads_per_cta() const
   {
      return m_block_dim.x * m_block_dim.y * m_block_dim.z;
   }

   dim3 get_grid_dim() const { return m_grid_dim; }
   dim3 get_cta_dim() const { return m_block_dim; }

   void increment_cta_id(unsigned core_id);
   dim3 get_next_cta_id(unsigned core_id) const;
   unsigned get_flattened_cta_id(dim3 tbid3) const
   {
     return tbid3.x + m_grid_dim.x*tbid3.y + m_grid_dim.x*m_grid_dim.y*tbid3.z;
   }


   dim3 get_dim3_id_kain(unsigned cta_id)
   {   
        dim3 tbid3;
        tbid3.z = cta_id / (m_grid_dim.x*m_grid_dim.y);
        tbid3.y = (cta_id - m_grid_dim.x*m_grid_dim.y*tbid3.z)/m_grid_dim.x;
        tbid3.x = cta_id - m_grid_dim.x*tbid3.y - m_grid_dim.x*m_grid_dim.y*tbid3.z;

        assert(get_flattened_cta_id(tbid3) == cta_id);

        return tbid3;
   }


   bool no_more_ctas_to_run() const
   {
       bool no_more = true;
       for(int i = 0; i < 4; i++)
            if(kain_m_next_cta[i].size() != 0)
                no_more = false;

      return (no_more)
             && m_to_be_migrated.empty() && m_to_be_switched.empty();

   }
   bool can_issue_new_cta(unsigned core_id) const;

   void increment_thread_id();
   dim3 get_next_thread_id_3d() const  { return m_next_tid; }
   unsigned get_next_thread_id() const
   {
      return m_next_tid.x + m_block_dim.x*m_next_tid.y + m_block_dim.x*m_block_dim.y*m_next_tid.z;
   }
   unsigned get_flattened_thread_id(dim3 tid3) const
   {
     return tid3.x + m_block_dim.x*tid3.y + m_block_dim.x*m_block_dim.y*tid3.z;
   }
   bool more_threads_in_cta() const
   {
      return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y && m_next_tid.x < m_block_dim.x;
   }
   unsigned get_uid() const { return m_uid; }
   std::string name() const;

   std::list<class ptx_thread_info *> &active_threads() { return m_active_threads; }
   class memory_space *get_param_memory() { return m_param_mem; }

   size_t executed_blocks() const;

   unsigned long long get_num_simulated_insts() const { return m_num_simulated_insts; }
   void inc_num_simulated_insts(unsigned active_count);
   void dec_num_simulated_insts(unsigned wasted_insts);

   ChildProcess* get_parent_process() { return m_parent; }
   unsigned get_pid() const { return m_parent->getID(); }
   void set_parent_process(ChildProcess* parent) { m_parent = parent; }
   void check_for_prev_stats();

   bool is_real_run() const { return m_real_run; }

   void set_init_max_cta_per_shader(unsigned _max_cta,unsigned shader_id);
   unsigned get_init_max_cta_per_shader() const { return init_max_cta_per_shader; }

   void set_max_cta_per_shader(const unsigned shader_id, const unsigned _max_cta);
   unsigned get_max_cta_per_shader(const unsigned shader_id) const { return cta_per_shader[shader_id]; }

   unsigned get_required_shaders() const;

   void inc_running_ctas() { ++m_running_ctas; }
   void dec_running_ctas() { assert(m_running_ctas > 0); --m_running_ctas; }

   void initialize_with_num_shaders(const unsigned num_shader, const unsigned num_cta_per_shader);
   void inc_scheduled_cta(const unsigned shader_id);
   void dec_scheduled_cta(const unsigned shader_id);

   unsigned get_scheduled_ctas(const unsigned shader_id) const { return scheduled_cta_per_shader[shader_id]; }

   void migrate_cta(dim3 migrating_cta_id) {
     m_to_be_migrated.push_back(migrating_cta_id);
   }

   void context_switch_cta(dim3 switching_cta_id, unsigned long long context_save_done_time, unsigned shader_id, unsigned hw_cta_id);
   bool is_context_switching_cta(const dim3& cta_id) const;

   // progress statistics
   const cta_stat_context_t& get_stat_context(unsigned sid, unsigned hw_cta_id) { return m_cta_statistics[sid][hw_cta_id]; }
   unsigned long long get_cta_num_insts() const { return m_cta_num_insts; }

   void start_cta(unsigned shader_id, unsigned hw_cta_id);
   void restart_cta(unsigned shader_id, unsigned hw_cta_id, const cta_stat_context_t& stored_stat);
   void finish_cta(unsigned shader_id, unsigned hw_cta_id);
   void stop_cta(unsigned shader_id, unsigned hw_cta_id);
   void inc_cta_inst(unsigned shader_id, unsigned hw_cta_id);

   // finding candidates for migration
   std::vector<unsigned> find_least_progressed_shaders(const std::vector<bool>& candidates, unsigned num_shaders) const;
   std::vector<unsigned> find_most_progressed_shaders(const std::vector<bool>& candidates, unsigned num_shaders) const;

   // finding candidates for hybrid preemption
   enum PreemptionTechnique {
     PREEMPTION_DRAIN,
     PREEMPTION_SWITCH,
     PREEMPTION_FLUSH,
     PREEMPTION_NUM
   };

   // chimera
   std::vector<std::pair<unsigned, PreemptionTechnique> > find_drain_switch_shaders(const std::vector<bool>& drain_candidates, const std::vector<bool>& switch_candidates, unsigned num_shaders) const;
   std::vector<std::pair<unsigned, PreemptionTechnique> > find_drain_flush_shaders(const std::vector<bool>& drain_candidates, const std::vector<bool>& flush_candidates, unsigned num_shaders) const;
   std::vector<std::pair<unsigned, PreemptionTechnique> > find_switch_flush_shaders(const std::vector<bool>& switch_candidates, const std::vector<bool>& flush_candidates, unsigned num_shaders) const;
   std::vector<std::pair<unsigned, PreemptionTechnique> > find_drain_switch_flush_shaders(const std::vector<bool>& drain_candidates, const std::vector<bool>& switch_candidates, const std::vector<bool>& flush_candidates, unsigned num_shaders) const;
   // fine-grain
   std::vector<preemption_info_t> find_fine_grain_preemptions(const std::vector<bool>& candidates, const std::vector<std::bitset<MAX_CTA_PER_SHADER> >& non_idempotent_ctas, unsigned num_shaders) const;

    void set_kain_stream_id(unsigned id) 
    {   
        kain_stream_id = id;    
    }   
    unsigned get_kain_stream_id()
    {   
        return kain_stream_id;  
    }
	unsigned long long kain_id;
//    std::map<unsigned, unsigned> kain_cta_cluster;

private:
    unsigned kain_stream_id;

   kernel_info_t( const kernel_info_t & ); // disable copy constructor
   void operator=( const kernel_info_t & ); // disable copy operator

   class function_info *m_kernel_entry;

   unsigned m_uid;
   static unsigned m_next_uid;

   dim3 m_grid_dim;
   dim3 m_block_dim;
   dim3 m_next_cta;
   dim3 m_next_tid;
   std::list<dim3> kain_m_next_cta[4];// 4-chiplet

   unsigned m_num_cores_running;

   std::list<class ptx_thread_info *> m_active_threads;
   class memory_space *m_param_mem;

   unsigned long long m_num_simulated_insts;
   ChildProcess* m_parent;
   bool m_real_run;
   unsigned init_max_cta_per_shader;

   unsigned m_running_ctas;

   std::vector<unsigned> cta_per_shader;
   std::vector<unsigned> scheduled_cta_per_shader;

   // cta stats
   std::vector<std::vector<cta_stat_context_t> > m_cta_statistics;

   // average stats
   unsigned long long m_cta_num_insts;
   double m_cta_avg_cycle;
   unsigned long long m_finished_ctas;

   // private helper functions
   void register_cta_num_insts(const unsigned long long curr_inst_count)
   {
     if (curr_inst_count > m_cta_num_insts) {
       m_cta_num_insts = curr_inst_count;
     }
   }

   unsigned long long get_executed_insts(unsigned sid, unsigned hw_tbid) const
   {
      return m_cta_statistics[sid][hw_tbid].get_executed_insts();
   }

   unsigned long long get_executed_cycles(unsigned sid, unsigned hw_tbid) const
   {
      return m_cta_statistics[sid][hw_tbid].get_executed_cycles();
   }

   unsigned long long get_executed_insts_for_shader(unsigned sid) const
   {
     assert(sid < m_cta_statistics.size());
     unsigned long long result = 0;
     const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[sid];
     for (unsigned i = 0; i < shader_stat.size(); ++i) {
       if (shader_stat[i].is_valid()) {
        result += shader_stat[i].get_executed_insts();
       }
     }
     return result;
   }

   unsigned long long get_executed_cycles_for_shader(unsigned sid) const
   {
     assert(sid < m_cta_statistics.size());
     unsigned long long result = 0;
     const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[sid];
     for (unsigned i = 0; i < shader_stat.size(); ++i) {
       if (shader_stat[i].is_valid()) {
         result += shader_stat[i].get_executed_cycles();
       }
     }
     return result;
   }

   unsigned long long num_valid_ctas(unsigned sid) const {
     assert(sid < m_cta_statistics.size());
     unsigned long long count = 0;
     const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[sid];
     for (unsigned i = 0; i < shader_stat.size(); ++i) {
       if (shader_stat[i].is_valid()) {
         ++count;
       }
     }
     return count;
   }

   unsigned long long get_switching_throughput_overhead(unsigned sid, unsigned hw_tbid) const
   {
     double ipc = 0.0;
     if (m_cta_num_insts == 0) {
        ipc = double(get_executed_insts_for_shader(sid)) / double(get_executed_cycles_for_shader(sid));
     } else {
        ipc = double(m_cta_num_insts) / m_cta_avg_cycle;
     }
     return (unsigned long long)(ipc * double(switching_overhead_per_cta) * 2);
   }

   bool can_estimate_drain() const
   {
     return m_finished_ctas != 0;
   }

   unsigned long long get_draining_latency(unsigned sid, unsigned hw_tbid) const
   {
     assert(can_estimate_drain());
     const unsigned long long avg_exec_cycle = (unsigned long long)m_cta_avg_cycle;
     const unsigned long long curr_exec_cycle = get_executed_cycles(sid, hw_tbid);
     if (avg_exec_cycle > curr_exec_cycle) {
       return avg_exec_cycle - curr_exec_cycle;
     }
     // if progressed more than average, assume it will be finished soon
     return 0;
   }

   unsigned long long get_draining_throughput_overhead(unsigned sid, unsigned hw_tbid) const
   {
     assert(can_estimate_drain());
     const std::vector<cta_stat_context_t>& shader_stat = m_cta_statistics[sid];

     unsigned long long max_exec_insts = 0;
     for (unsigned i = 0; i < shader_stat.size(); ++i) {
       if (shader_stat[i].is_valid()) {
         max_exec_insts = std::max(max_exec_insts, shader_stat[i].get_executed_insts());
       }
     }

     return max_exec_insts - shader_stat[hw_tbid].get_executed_insts();
   }

   unsigned long long get_max_remaining_cycle_for_shader(unsigned shader_id) const;
   unsigned long long get_switching_overhead_latency_for_shader(unsigned shader_id) const;

   unsigned long long get_draining_throughput_overhead_for_shader(unsigned shader_id) const;
   unsigned long long get_switching_throughput_overhead_for_shader(unsigned shader_id) const;

public:
   enum limitCause {
     LIMITED_BY_THREADS,
     LIMITED_BY_SHMEM,
     LIMITED_BY_REGS,
     LIMITED_BY_CTA,
     LIMITED_BY_MSHR,
     NUM_LIMIT_CAUSE
   };

   limitCause get_max_cta_reason() const { return max_cta_per_shader_reason; }
   // HACK WARNING
   // we make this const due to shader implementation
   void set_max_cta_reason(limitCause reason) const { max_cta_per_shader_reason = reason; }

   bool has_atomic() const { return m_has_atomic; }
   void set_has_atomic()   { m_has_atomic = true; }

   bool overwrites_input() const { return m_overwrites_input; }
   void set_overwrites_input()   { m_overwrites_input = true; }

   void set_switching_overhead(unsigned long long overhead) { switching_overhead_per_cta = overhead; }

private:
   // HACK WARNING
   // intentionally made mutable, but semantically should not be.
   mutable limitCause max_cta_per_shader_reason;

   bool m_has_atomic;
   bool m_overwrites_input;

   std::vector<dim3> m_to_be_migrated;
   std::priority_queue<timed_dim3, std::vector<timed_dim3>, timed_dim3_comparison> m_to_be_switched;

   unsigned long long switching_overhead_per_cta;
};

#endif // KERNEL_INFO_T_H

