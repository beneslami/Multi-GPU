#ifndef __MK_SCHEDULER_H__
#define __MK_SCHEDULER_H__

#include <list>
#include <vector>
#include <bitset>

#include "named_info.h"
#include "../../common/hard_consts.h"
#include "../../common/preemption_info.h"
#include "../../common/kernel_info_t.h"

class kernel_info_t;

extern bool Stream1_SM[384];
extern bool Stream2_SM[192];
extern bool Stream3_SM[192];
extern bool Stream4_SM[192];



//void KAIN_check_whether_can_calculate_IPC()// first 0-3, 8-11 SM are same kernel; second, 5K cycles
//{


//}

//void KAIN_end_profile_cal_IPC_re-schedule()
//{


//}

// abstract class for multikernel scheduler
class MKScheduler {
public:
  // must contain SchedulerType
  static MKScheduler* Create(const SchedulerInfo& info);

public:
  MKScheduler() : num_SMs(0)
  {}

  virtual ~MKScheduler()
  {}

public:
  // these are different for each scheduler
  virtual kernel_info_t* next_thread_block_to_schedule() = 0;

  kernel_info_t * kain_next_thread_block_to_schedule(unsigned shader_id)
  {
          kernel_info_t* result = NULL;
          assert(kernels.size() <= 4);// kain add this, there are only 4 streams now
          assert(shader_id <= 384);// kain add this, there are only 2 streams now


          for (std::list<kernel_info_t*>::iterator it = kernels.begin(), it_end = kernels.end();
               it != it_end; ++it) {
              //result = *it;
              kernel_info_t* tmp= *it;
                  
			  if(tmp->get_kain_stream_id() == 1)//stream 1
              {
                    if(Stream1_SM[shader_id] == true)
                        result = tmp;
              }   
			  else if(tmp->get_kain_stream_id() == 2)//stream 2
              {   
                    if(Stream2_SM[shader_id] == true)
                        result = tmp;
              }
			  else if(tmp->get_kain_stream_id() == 3)//stream 3
              {   
                    if(Stream3_SM[shader_id] == true)
                        result = tmp;
              }
			  else if(tmp->get_kain_stream_id() == 4)//stream 4
              {   
                    if(Stream4_SM[shader_id] == true)
                        result = tmp;
              }
              else
                    assert(0);
          }
//		  printf("assign kernel for shader %d\n",shader_id);
//		  fflush(stdout);
          return result;
  }




  virtual void update_scheduler(const SchedulerUpdateInfo& info) = 0;
  virtual void inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id) = 0;
  virtual void dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id) = 0;

  // whether a kernel needs to reduce its SM (NULL if not)
  virtual std::vector<kernel_info_t*> check_for_removal() const = 0;
  virtual std::vector<kernel_info_t*> check_for_cancel_removal() const = 0;
  // number of shaders to be removed or canceled
  virtual unsigned get_num_remove_required(kernel_info_t* kernel) const = 0;
  virtual unsigned get_num_cancel_remove_required(kernel_info_t* kernel) const = 0;
  // find shaders to remove or cancel
  virtual std::vector<preemption_info_t> find_shaders_to_remove(kernel_info_t* kernel, unsigned num_remove) = 0;
  virtual std::vector<unsigned> find_shaders_to_cancel(kernel_info_t* kernel, unsigned num_cancel) = 0;
  // shaders that loads context switch
  virtual void shader_loads_context(kernel_info_t* kernel, unsigned shader_id) {}
  virtual void shader_finishes_loading_context(kernel_info_t* kernel, unsigned shader_id) {}

  // each scheduler can add more functionality when adding/removing kernels
  virtual void add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader) { kernels.push_back(kernel); }
  virtual void remove_kernel(kernel_info_t* kernel)                           { kernels.remove(kernel); }

  // start/stop/finish CTA
  virtual void start_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) = 0;
  virtual void stop_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) = 0;
  virtual void finish_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) = 0;
  // CTA executes specific instructions
  virtual void CTA_executes_atomic(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) = 0;
  virtual void CTA_overwrites_input(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) = 0;

  // each scheduler can perform cycle by cycle action if necessary
  virtual void core_cycle(const std::vector<unsigned> & scheduled_ctas)       {}
  virtual void l2_cache_cycle()                                               {}

  // deal with statistics if exists
  virtual void clear_statistics() {}
  virtual void print_statistics() {}

  // common methods
  virtual void set_number_of_SMs(unsigned num_shader) { num_SMs = num_shader; }

protected:
  unsigned num_SMs;
  std::list<kernel_info_t*> kernels;
};

// common schduler that statically changes number of SMs for multikernels
class StaticScheduler : public MKScheduler {
public:
  StaticScheduler()  {}
  virtual ~StaticScheduler() {}

public:
  // whether a kernel needs to reduce its SM (NULL if not)
  virtual std::vector<kernel_info_t*> check_for_removal() const { return std::vector<kernel_info_t*>(); }
  virtual std::vector<kernel_info_t*> check_for_cancel_removal() const { return std::vector<kernel_info_t*>(); }
  // number of shaders to be removed or canceled
  virtual unsigned get_num_remove_required(kernel_info_t* kernel) const { return 0; }
  virtual unsigned get_num_cancel_remove_required(kernel_info_t* kernel) const { return 0; }
  // find shaders to remove or cancel
  virtual std::vector<preemption_info_t> find_shaders_to_remove(kernel_info_t* kernel, unsigned num_remove) { return std::vector<preemption_info_t>(); }
  virtual std::vector<unsigned> find_shaders_to_cancel(kernel_info_t* kernel, unsigned num_cancel) { return std::vector<unsigned>(); }
  // start/finish CTA
  virtual void start_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)  {}
  virtual void stop_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)   {}
  virtual void finish_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) {}
  // CTA executes specific instructions
  virtual void CTA_executes_atomic(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id)  {}
  virtual void CTA_overwrites_input(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id) {}
};

// common schduler that dynamically changes number of SMs for multikernels
class DynamicScheduler : public MKScheduler {
public:
  DynamicScheduler(const std::string& _preempt, const std::string& _progress);
  virtual ~DynamicScheduler() {}

public:
  virtual kernel_info_t* next_thread_block_to_schedule();
  virtual void update_scheduler(const SchedulerUpdateInfo& info) = 0;
  virtual void inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id);
  virtual void dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id);

  virtual std::vector<kernel_info_t*> check_for_removal() const;
  virtual std::vector<kernel_info_t*> check_for_cancel_removal() const;
  virtual unsigned get_num_remove_required(kernel_info_t* kernel) const;
  virtual unsigned get_num_cancel_remove_required(kernel_info_t* kernel) const;

  virtual std::vector<preemption_info_t> find_shaders_to_remove(kernel_info_t* kernel, unsigned num_remove);
  virtual std::vector<unsigned> find_shaders_to_cancel(kernel_info_t* kernel, unsigned num_cancel);
  virtual void shader_loads_context(kernel_info_t* kernel, unsigned shader_id);
  virtual void shader_finishes_loading_context(kernel_info_t* kernel, unsigned shader_id);

  virtual void add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader);
  virtual void remove_kernel(kernel_info_t* kernel);

  virtual void start_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id);
  virtual void stop_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id);
  virtual void finish_CTA(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id);
  virtual void CTA_executes_atomic(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id);
  virtual void CTA_overwrites_input(kernel_info_t* kernel, const unsigned shader_id, const unsigned cta_id);

protected:
  class SM_info {
  public:
    SM_info()
      : maximum_alloc(0), holding_shaders(0), expected_shaders(0), removing_shaders(0)
    {}

    struct remove_status_st {
      enum RemoveStatus {
        REMOVE_DRAINING,
        REMOVE_SWITCHING,
        REMOVE_FLUSHING,
        REMOVE_NUM
      };

      RemoveStatus status;
      bool is_being_removed;

      remove_status_st()
        : status(REMOVE_NUM), is_being_removed(false)
      {}

      void reset() { status = REMOVE_NUM; is_being_removed = false; }
      bool is_draining() const  { return status == REMOVE_DRAINING;  }
      bool is_context_saving() const { return status == REMOVE_SWITCHING; }
      bool is_flushing() const  { return status == REMOVE_FLUSHING;  }
      void make_drain()  { status = REMOVE_DRAINING;  is_being_removed = true; }
      void make_switch() { status = REMOVE_SWITCHING; is_being_removed = true; }
      void make_flush()  { status = REMOVE_FLUSHING;  is_being_removed = true; }
    };

    void initialize(const unsigned _num_shaders)
    {
      occupied.resize(_num_shaders, false);
      context_loading.resize(_num_shaders, false);
      to_remove.resize(_num_shaders);
      num_shaders = _num_shaders;
      maximum_alloc = _num_shaders;
      cta_running.resize(_num_shaders);
      cta_non_idempotent.resize(_num_shaders);
      for (unsigned i = 0; i < _num_shaders; ++i) {
        cta_running[i].reset();
        cta_non_idempotent[i].reset();
      }
    }

    // accessor
    bool is_occupying(const unsigned linear_shader_id) const      { return occupied[linear_shader_id]; }
    bool is_to_be_removed(const unsigned linear_shader_id) const  { return to_remove[linear_shader_id].is_being_removed; }
    bool is_draining(const unsigned linear_shader_id) const       { return to_remove[linear_shader_id].is_draining(); }
    bool is_context_saving(const unsigned linear_shader_id) const { return to_remove[linear_shader_id].is_context_saving(); }
    bool is_flushing(const unsigned linear_shader_id) const       { return to_remove[linear_shader_id].is_flushing(); }
    bool is_cta_running(unsigned sid, unsigned hw_cta_id) const   { return cta_running[sid][hw_cta_id]; }

    bool is_context_loading(const unsigned linear_shader_id) const;
    void make_context_load(const unsigned linear_shader_id);
    void make_context_load_done(const unsigned linear_shader_id);

    const std::vector<bool>& get_occupied_vector() const
    {
      return occupied;
    }

    const std::vector<bool> get_drainable_vector() const
    {
      std::vector<bool> result(occupied);
      for (unsigned i = 0; i < num_shaders; ++i) {
        if (context_loading[i]) {
          result[i] = false;
        }
      }
      return result;
    }

    const std::vector<bool> get_switchable_vector() const
    {
      return get_drainable_vector();
    }

    const std::vector<bool> get_flushable_vector() const
    {
      std::vector<bool> result(occupied);
      for (unsigned i = 0; i < num_shaders; ++i) {
        if (cta_non_idempotent[i].any() || context_loading[i]) {
          result[i] = false;
        }
      }
      return result;
    }

    const std::vector<bool> get_idempotent_occupied_vector() const
    {
      std::vector<bool> result(occupied);
      for (unsigned i = 0; i < num_shaders; ++i) {
        if (cta_non_idempotent[i].any()) {
          result[i] = false;
        }
      }
      return result;
    }

    unsigned num_holding_shader() const   { return holding_shaders; }
    unsigned num_expected_shader() const  { return expected_shaders; }
    unsigned num_removing_shader() const  { return removing_shaders; }

    // modifier
    void add_shader(unsigned linear_shader_id);
    void remove_shader(unsigned linear_shader_id);
    void preempt_shader(unsigned linear_shader_id);
    void set_shader_to_drain(const unsigned linear_shader_id);
    void set_shader_to_switch(const unsigned linear_shader_id);
    void set_shader_to_flush(const unsigned linear_shader_id);
    void set_upper_bound(const unsigned bound);
    void set_initial_alloc(const unsigned alloc);
    void cancel_removal(const unsigned linear_shader_id);

    void cta_starts_running(const unsigned shader_id, const unsigned cta_id);
    void cta_stops_running(const unsigned shader_id, const unsigned cta_id);
    void cta_finishes_running(const unsigned shader_id, const unsigned cta_id);
    void cta_passes_non_idempotent_region(const unsigned shader_id, const unsigned cta_id);

    // accessor
    const std::vector<std::bitset<MAX_CTA_PER_SHADER> >& get_non_idempotent_vector() const  { return cta_non_idempotent; }

  private:
    std::vector<bool> occupied;
    std::vector<bool> context_loading;
    std::vector<remove_status_st> to_remove;

    unsigned maximum_alloc;
    unsigned holding_shaders;
    unsigned expected_shaders;
    unsigned removing_shaders;

    unsigned num_shaders;

    std::vector<std::bitset<MAX_CTA_PER_SHADER> > cta_running;
    std::vector<std::bitset<MAX_CTA_PER_SHADER> > cta_non_idempotent;
  };

  std::map<kernel_info_t*, SM_info> SM_infos;

protected:
  enum PreemptionPolicy {
    PREEMPT_DRAIN_ONLY,
    PREEMPT_SWITCH_ONLY,
    PREEMPT_FLUSH_ONLY,
    PREEMPT_DRAIN_SWITCH,
    PREEMPT_DRAIN_FLUSH,
    PREEMPT_SWITCH_FLUSH,
    PREEMPT_DRAIN_SWITCH_FLUSH,
    PREEMPT_FINE_GRAIN,
    PREEMPT_NUM
  };

  enum ProgressPolicy {
    PROGRESS_INST,
    PROGRESS_CYCLE,
    PROGRESS_NUM
  };

  PreemptionPolicy preempt_policy;
  ProgressPolicy progress_policy;

  bool check_progress_in_inst() const { return progress_policy == PROGRESS_INST; }
};

#endif // __MK_SCHEDULER_H__

