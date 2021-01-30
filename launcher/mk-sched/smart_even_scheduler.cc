#include <cassert>

#include "../../common/kernel_info_t.h"
#include "smart_even_scheduler.h"

extern unsigned long long get_curr_cycle();

void
SmartEvenScheduler::update_scheduler(const SchedulerUpdateInfo& info)
{
  if (info.has("AdjustSMs")) {
    adjust_upper_bounds();
  }
}

void
SmartEvenScheduler::add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader)
{
  DynamicScheduler::add_kernel(kernel, max_cta_per_shader);
  adjust_upper_bounds();
}

void
SmartEvenScheduler::remove_kernel(kernel_info_t* kernel)
{
  DynamicScheduler::remove_kernel(kernel);
  adjust_upper_bounds();
}

void
SmartEvenScheduler::adjust_upper_bounds()
{
  const unsigned num_kernels = SM_infos.size();
  const unsigned even_split = num_SMs / num_kernels;
  int remaining_SMs = num_SMs - (even_split * num_kernels);

  std::vector<kernel_info_t*> non_thread_block_bound_kernels;
  std::map<kernel_info_t*, unsigned> new_allocs;
  for (std::map<kernel_info_t*, SM_info>::iterator it = SM_infos.begin(), it_end = SM_infos.end();
       it != it_end; ++it) {
    kernel_info_t* kernel = it->first;
    unsigned required_shaders = kernel->get_required_shaders();

    if (even_split >= required_shaders) {
      // thread block bound!!
      remaining_SMs += even_split - required_shaders;
      new_allocs[kernel] = required_shaders;
      //it->second.set_upper_bound(required_shaders);
    } else {
      new_allocs[kernel] = even_split;
      non_thread_block_bound_kernels.push_back(kernel);
    }
  }

  while (remaining_SMs > 0 && !non_thread_block_bound_kernels.empty()) {
    bool no_kernel_to_allocate = true;
    for (std::vector<kernel_info_t*>::const_iterator it = non_thread_block_bound_kernels.begin(), it_end = non_thread_block_bound_kernels.end();
         it != it_end; ++it) {
      kernel_info_t* kernel = *it;

      // not thread block bound
      if (kernel->get_required_shaders() > new_allocs[kernel]) {
        ++new_allocs[kernel];
        --remaining_SMs;
        no_kernel_to_allocate = false;
      }
    }

    if (no_kernel_to_allocate) {
      break;
    }
  }

  for (std::map<kernel_info_t*, SM_info>::iterator it = SM_infos.begin(), it_end = SM_infos.end();
       it != it_end; ++it) {
    kernel_info_t* kernel = it->first;
    if (new_allocs[kernel] != it->second.num_expected_shader()) {
      printf("GPGPU-Sim Smart: Process %d, kernel %s, sets SMs to %d @(%llu)\n", kernel->get_parent_process()->getID(), kernel->name().c_str(), new_allocs[kernel], get_curr_cycle());
    }
    it->second.set_upper_bound(new_allocs[kernel]);
  }
}

