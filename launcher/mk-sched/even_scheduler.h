#ifndef __EVEN_SCHEDULER_H__
#define __EVEN_SCHEDULER_H__

#include "mk_scheduler.h"

// evenly splits GPU for multikernels
class EvenScheduler : public StaticScheduler {
public:
  EvenScheduler()  {}
  virtual ~EvenScheduler() {}

public:
  virtual kernel_info_t* next_thread_block_to_schedule();
  virtual void update_scheduler(const SchedulerUpdateInfo& info);
  virtual void inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id);
  virtual void dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id);

  virtual void add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader);
  virtual void remove_kernel(kernel_info_t* kernel);

private:
  std::map<kernel_info_t*, unsigned> expected_SMs;
  std::map<kernel_info_t*, unsigned> holding_SMs;

  void adjust_expected_SMs();
};

#endif // __EVEN_SCHEDULER_H__

