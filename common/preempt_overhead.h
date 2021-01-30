#ifndef PREEMPT_OVERHEAD_H_
#define PREEMPT_OVERHEAD_H_

#include "kernel_info_t.h"
#include <map>
#include <queue>

// this will be used only within finding shaders to remove
class preempt_overhead_item {
public:
  unsigned shader_id;
  unsigned long long throughput_overhead;
  unsigned long long latency_overhead;
  kernel_info_t::PreemptionTechnique preempt;

  preempt_overhead_item(unsigned _shader_id, unsigned long long _throughput_overhead, unsigned long long _latency_overhead, kernel_info_t::PreemptionTechnique _preempt)
    : shader_id(_shader_id),
      throughput_overhead(_throughput_overhead),
      latency_overhead(_latency_overhead),
      preempt(_preempt)
  {}
};

class cta_preempt_overhead {
public:
  unsigned hw_cta_id;
  unsigned long long throughput_overhead;
  unsigned long long latency_overhead;
  kernel_info_t::PreemptionTechnique preempt;

  cta_preempt_overhead(unsigned _hw_cta_id, unsigned long long _throughput_overhead, unsigned long long _latency_overhead, kernel_info_t::PreemptionTechnique _preempt)
    : hw_cta_id(_hw_cta_id),
      throughput_overhead(_throughput_overhead),
      latency_overhead(_latency_overhead),
      preempt(_preempt)
  {}
};

class shader_preempt_overhead {
public:
  unsigned shader_id;
  unsigned long long throughput_overhead;
  unsigned long long latency_overhead;
  std::map<unsigned, kernel_info_t::PreemptionTechnique> preemption_techniques;

  shader_preempt_overhead(unsigned _shader_id)
    : shader_id(_shader_id),
      throughput_overhead(0),
      latency_overhead(0),
      max_drain_latency(0)
  {}

  void add_cta_with_preemption(const cta_preempt_overhead& cta_overhead)
  {
      // draining latency overhead is overlapped
      if (cta_overhead.preempt == kernel_info_t::PREEMPTION_DRAIN) {
          if (cta_overhead.latency_overhead > max_drain_latency) {
              latency_overhead += cta_overhead.latency_overhead - max_drain_latency;
              max_drain_latency = cta_overhead.latency_overhead;
          }
      } else {
          latency_overhead += cta_overhead.latency_overhead;
      }
      throughput_overhead += cta_overhead.throughput_overhead;
      assert(!cta_exists(cta_overhead.hw_cta_id));
      preemption_techniques[cta_overhead.hw_cta_id] = cta_overhead.preempt;
  }

  bool cta_exists(unsigned hw_cta_id) const
  {
      return preemption_techniques.find(hw_cta_id) != preemption_techniques.end();
  }

  bool meets_latency_constraint(const cta_preempt_overhead& cta_overhead) const;

  unsigned size() const
  {
      return preemption_techniques.size();
  }

private:
  unsigned long long max_drain_latency;
};

class preempt_throughput_overhead_comparison {
public:
  preempt_throughput_overhead_comparison() {}
  bool operator() (const preempt_overhead_item& lhs, const preempt_overhead_item& rhs) const
  {
    if (lhs.throughput_overhead == rhs.throughput_overhead) {
      if (lhs.latency_overhead == rhs.latency_overhead) {
        // prefer draining, switching, flushing
        return lhs.preempt > rhs.preempt;
      }
      return lhs.latency_overhead > rhs.latency_overhead;
    }
    return lhs.throughput_overhead > rhs.throughput_overhead;
  }
};

class cta_preempt_overhead_compare {
public:
  cta_preempt_overhead_compare() {}
  bool operator() (const cta_preempt_overhead& lhs, const cta_preempt_overhead& rhs) const
  {
    if (lhs.throughput_overhead == rhs.throughput_overhead) {
      if (lhs.latency_overhead == rhs.latency_overhead) {
        // prefer draining, switching, flushing
        return lhs.preempt > rhs.preempt;
      }
      return lhs.latency_overhead > rhs.latency_overhead;
    }
    return lhs.throughput_overhead > rhs.throughput_overhead;
  }
};

class shader_preempt_overhead_compare {
public:
  shader_preempt_overhead_compare() {}
  bool operator() (const shader_preempt_overhead& lhs, const shader_preempt_overhead& rhs) const
  {
    if (lhs.throughput_overhead == rhs.throughput_overhead) {
      if (lhs.latency_overhead == rhs.latency_overhead) {
        // prefer draining, switching, flushing
        return lhs.preemption_techniques.size() > rhs.preemption_techniques.size();
      }
      return lhs.latency_overhead > rhs.latency_overhead;
    }
    return lhs.throughput_overhead > rhs.throughput_overhead;
  }
};

typedef std::priority_queue<preempt_overhead_item, std::vector<preempt_overhead_item>, preempt_throughput_overhead_comparison> sort_throughput_queue;
typedef std::priority_queue<cta_preempt_overhead , std::vector<cta_preempt_overhead>, cta_preempt_overhead_compare> cta_sort;
typedef std::priority_queue<shader_preempt_overhead, std::vector<shader_preempt_overhead>, shader_preempt_overhead_compare> shader_sort;

std::vector<std::pair<unsigned, kernel_info_t::PreemptionTechnique> >
select_candidates(unsigned num_shaders, sort_throughput_queue& sorted_candidates);

std::vector<preemption_info_t>
select_candidates(unsigned num_shaders, shader_sort& sorted_candidates);

#endif // PREEMPT_OVERHEAD_H_

