#ifndef __SMART_EVEN_SCHEDULER_H__
#define __SMART_EVEN_SCHEDULER_H__

#include "mk_scheduler.h"

// evenly splits GPU for multikernels
class SmartEvenScheduler : public DynamicScheduler {
public:
  SmartEvenScheduler(const std::string& _preempt, const std::string& _progress)
    : DynamicScheduler(_preempt, _progress)
  {}
  virtual ~SmartEvenScheduler() {}

public:
  virtual void update_scheduler(const SchedulerUpdateInfo& info);

  virtual void add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader);
  virtual void remove_kernel(kernel_info_t* kernel);

private:
  void adjust_upper_bounds();
};

#endif // __SMART_EVEN_SCHEDULER_H__

