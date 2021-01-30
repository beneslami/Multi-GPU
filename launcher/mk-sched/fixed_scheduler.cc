#include <cassert>

#include "../../common/kernel_info_t.h"
#include "fixed_scheduler.h"

FixedScheduler::FixedScheduler(const std::list<unsigned>& allocs)
  : StaticScheduler(), free_list(allocs)
{
}

kernel_info_t*
FixedScheduler::next_thread_block_to_schedule()
{
  kernel_info_t* result = NULL;
  int max_diff = 0;

  for (std::list<kernel_info_t*>::iterator it = kernels.begin(), it_end = kernels.end();
       it != it_end; ++it) {
    int curr_diff = allocated[*it].alloc - allocated[*it].holding;
    if (curr_diff > max_diff && !(*it)->no_more_ctas_to_run()) {
      result = *it;
      max_diff = curr_diff;
    }
  }

  return result;
}

void
FixedScheduler::update_scheduler(const SchedulerUpdateInfo& info)
{
}

void
FixedScheduler::inc_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id)
{
  assert(allocated.find(kernel) != allocated.end());
  assert(allocated[kernel].holding < allocated[kernel].alloc);
  allocated[kernel].holding += 1;
}

void
FixedScheduler::dec_SM_for_kernel(kernel_info_t* kernel, unsigned linear_shader_id)
{
  assert(allocated.find(kernel) != allocated.end());
  assert(allocated[kernel].holding > 0);
  allocated[kernel].holding -= 1;
}

void
FixedScheduler::add_kernel(kernel_info_t* kernel, unsigned max_cta_per_shader)
{
  assert(allocated.find(kernel) == allocated.end());
  printf("GPGPU-Sim FixedScheduler: Process %d adds kernel @ 0x%llx..\n", kernel->get_parent_process()->getID(), (unsigned long long) kernel);

  // kernel is launched in reverse order
  // here we pop from the back so that it matches with the given allocation
  allocated[kernel] = SM_alloc(free_list.back());
  free_list.pop_back();

  MKScheduler::add_kernel(kernel, max_cta_per_shader);
}

void
FixedScheduler::remove_kernel(kernel_info_t* kernel)
{
  assert(allocated.find(kernel) != allocated.end());
  printf("GPGPU-Sim FixedScheduler: Process %d removes kernel @ 0x%llx..\n", kernel->get_parent_process()->getID(), (unsigned long long) kernel);

  // if kernel is removed, same application will launch a new kernel.
  // therefore, it should be pushed at back
  free_list.push_back(allocated[kernel].alloc);
  allocated.erase(kernel);

  MKScheduler::remove_kernel(kernel);
}

void
FixedScheduler::set_number_of_SMs(unsigned num_shader)
{
  StaticScheduler::set_number_of_SMs(num_shader);

  unsigned sumAllocs = 0;
  for (std::list<unsigned>::iterator it = free_list.begin(), it_end = free_list.end();
       it != it_end; ++it) {
    sumAllocs += (*it);
  }

//  assert(num_SMs == sumAllocs);
}

