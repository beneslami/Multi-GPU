#include <cassert>
#include "cta_stat_context.h"

extern unsigned long long get_curr_cycle();

cta_stat_context_t::cta_stat_context_t()
{
  clear_context();
}

void
cta_stat_context_t::save_context(const cta_stat_context_t& to_save)
{
  assert(to_save.m_valid);
  assert(!to_save.m_saved);
  m_valid = to_save.m_valid;
  m_saved = true;
  m_executed_insts = to_save.m_executed_insts;
  m_executed_cycles = get_curr_cycle() - to_save.m_started_cycles;
}

void
cta_stat_context_t::load_context(const cta_stat_context_t& to_load)
{
  assert(to_load.m_valid);
  assert(to_load.m_saved);
  assert(!m_valid);
  assert(!m_saved);
  m_valid = to_load.m_valid;
  m_saved = false;
  m_executed_insts = to_load.m_executed_insts;
  m_started_cycles = get_curr_cycle() - to_load.m_executed_cycles;
}

void
cta_stat_context_t::clear_context()
{
  m_valid = false;
  m_saved = false;
  m_executed_insts = 0;
  m_started_cycles = 0;
}

void
cta_stat_context_t::start_stat()
{
  assert(!m_valid);
  assert(!m_saved);
  m_valid = true;
  m_saved = false;
  m_executed_insts = 0;
  m_started_cycles = get_curr_cycle();
}

unsigned long long
cta_stat_context_t::get_executed_insts() const
{
  assert(m_valid);
  assert(!m_saved);
  return m_executed_insts;
}

unsigned long long
cta_stat_context_t::get_executed_cycles() const
{
  assert(m_valid);
  assert(!m_saved);
  assert(get_curr_cycle() >= m_started_cycles);

  return get_curr_cycle() - m_started_cycles;
}

unsigned long long
cta_stat_context_t::get_started_cycle() const
{
  assert(m_valid);
  assert(!m_saved);
  assert(get_curr_cycle() >= m_started_cycles);

  return m_started_cycles;
}

