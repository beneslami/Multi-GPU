#include <set>
#include "preempt_overhead.h"

// hard-coded limit for latency
extern unsigned long long LATENCY_LIMIT;

bool
shader_preempt_overhead::meets_latency_constraint(const cta_preempt_overhead& cta_overhead) const
{
  // if draining, latency overhead can be overlapped
  unsigned long long curr_latency_overhead = cta_overhead.latency_overhead;
  if (cta_overhead.preempt == kernel_info_t::PREEMPTION_DRAIN) {
    if (cta_overhead.latency_overhead > max_drain_latency) {
      curr_latency_overhead -= max_drain_latency;
    } else {
      curr_latency_overhead = 0;
    }
  }

  // remaining latency constraint >= cta latency overhead
  return (LATENCY_LIMIT - latency_overhead) >= curr_latency_overhead;
}

std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >
select_candidates(unsigned num_shaders, sort_throughput_queue& sorted_candidates)
{
  std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> > results;
  std::set<unsigned> selected_shaders;
  for (unsigned i = 0; i < num_shaders; ++i) {
    if (sorted_candidates.empty()) {
      break;
    }

    while (!sorted_candidates.empty()) {
      const preempt_overhead_item& item = sorted_candidates.top();
      if (selected_shaders.find(item.shader_id) == selected_shaders.end() && item.latency_overhead <= LATENCY_LIMIT) {
        results.push_back(std::make_pair(item.shader_id, item.preempt));
        selected_shaders.insert(item.shader_id);
        sorted_candidates.pop();
        break;
      }
      sorted_candidates.pop();
    }
  }

  return results;
}

std::vector<preemption_info_t>
select_candidates(unsigned num_shaders, shader_sort& sorted_candidates)
{
  std::vector<preemption_info_t> results;
  std::set<unsigned> selected_shaders;
  shader_sort remainders;
  for (unsigned i = 0; i < num_shaders; ++i) {
    if (sorted_candidates.empty()) {
      break;
    }

    while (!sorted_candidates.empty()) {
      const shader_preempt_overhead& item = sorted_candidates.top();
      if (selected_shaders.find(item.shader_id) == selected_shaders.end() && item.latency_overhead <= LATENCY_LIMIT) {
        results.push_back(preemption_info_t(item));
        selected_shaders.insert(item.shader_id);
        sorted_candidates.pop();
        break;
      }
      remainders.push(item);
      sorted_candidates.pop();
    }
  }

  while (results.size() < num_shaders && !remainders.empty()) {
    const shader_preempt_overhead& item = remainders.top();
    if (selected_shaders.find(item.shader_id) == selected_shaders.end() && item.latency_overhead <= 2 * LATENCY_LIMIT) {
      results.push_back(preemption_info_t(item));
      selected_shaders.insert(item.shader_id);
    }
    remainders.pop();
  }

  return results;
}

