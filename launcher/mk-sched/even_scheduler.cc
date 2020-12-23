#include <cassert>

#include "../../common/kernel_info_t.h"
#include "even_scheduler.h"

kernel_info_t*
EvenScheduler::next_thread_block_to_schedule()
{
  kernel_info_t* result = NULL;
  int max_diff = 0;

  for (std::list<kernel_info_t*>::iterator it = kernels.begin(), it_end = kernels.end();
       it != it_end; ++it) {
    int curr_diff = expected_SMs[*it] - holding_SMs[*it];
    if (curr_diff > max_diff && !(*it)->no_more_ctas_to_run()) {
      result = *it;
      max_diff = curr_diff;
    }
  }

  return result;
}

void
EvenScheduler::update_scheduler(const SchedulerUpdateInfo& info)
{
}

void
EvenScheduler::inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id)
{
  assert(holding_SMs.find(kernel) != holding_SMs.end());
  holding_SMs[kernel] += 1;
}

void
EvenScheduler::dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id)
{
  assert(holding_SMs.find(kernel) != holding_SMs.end());
  holding_SMs[kernel] -= 1;
}

void
EvenScheduler::add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader)
{
  assert(expected_SMs.find(kernel) == expected_SMs.end());

  expected_SMs[kernel] = 0;
  holding_SMs[kernel] = 0;
  MKScheduler::add_kernel(kernel, max_cta_per_shader);

  adjust_expected_SMs();
}

void
EvenScheduler::remove_kernel(kernel_info_t* kernel)
{
  assert(expected_SMs.find(kernel) != expected_SMs.end());

  expected_SMs.erase(kernel);
  holding_SMs.erase(kernel);
  MKScheduler::remove_kernel(kernel);

  adjust_expected_SMs();
}

void
EvenScheduler::adjust_expected_SMs()
{
  const unsigned num_kernels = expected_SMs.size();
  assert(num_kernels == holding_SMs.size());
  assert(num_kernels == kernels.size());

  if (expected_SMs.empty()) {
    return;
  }

  const unsigned bottom = num_SMs / num_kernels;
  int remaining = num_SMs - (bottom * num_kernels);
  for (std::map<kernel_info_t*, unsigned>::iterator it = expected_SMs.begin(), it_end = expected_SMs.end();
       it != it_end; ++it) {
    it->second = bottom;
    if (remaining > 0) {
      it->second += 1;
      --remaining;
    }
  }
}

