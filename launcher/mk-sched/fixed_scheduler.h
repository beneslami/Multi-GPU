#ifndef __FIXED_SCHEDULER_H__
#define __FIXED_SCHEDULER_H__

#include "mk_scheduler.h"

// evenly splits GPU for multikernels
class FixedScheduler : public StaticScheduler {
public:
  FixedScheduler(const std::list<unsigned>& allocs);
  virtual ~FixedScheduler() {}

public:
  virtual kernel_info_t* next_thread_block_to_schedule();
  virtual void update_scheduler(const SchedulerUpdateInfo& info);
  virtual void inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id);
  virtual void dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id);

  virtual void add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader);
  virtual void remove_kernel(kernel_info_t* kernel);

  virtual void set_number_of_SMs(unsigned num_shader);

private:
  struct SM_alloc {
    unsigned holding;
    unsigned alloc;

    SM_alloc(unsigned _alloc = 0)
      : holding(0), alloc(_alloc)
    {}
  };

  std::list<unsigned> free_list;
  std::map<kernel_info_t*, SM_alloc> allocated;
};

#endif // __FIXED_SCHEDULER_H__

