#ifndef __STATS_H__
#define __STATS_H__

#include <string>
#include <cassert>
#include <stdint.h>

namespace Stats {

unsigned long long get_curr_cycle();

// base class for statistics
class StatBase {
public:
  StatBase()  {}
  ~StatBase() {}

public:
  void name(const std::string &name)
  {
    m_name = name;
  }

  const std::string &name() const { return m_name; }

private:
  std::string m_name;
};

// abstract storage class for Scalar
template<typename T>
class ScalarStorage {
public:
  ScalarStorage()  {}
  ~ScalarStorage() {}

  virtual void reinit() = 0;

public:
  // modifier
  virtual void set(uint64_t val) = 0;
  virtual void inc(uint64_t val) = 0;
  virtual void dec(uint64_t val) = 0;

public:
  // accessor
  virtual void finish()    = 0;
  virtual T result() const = 0;
};

// storage class for Scalar that stores Maximum value
class MaxScalarStorage : ScalarStorage<uint64_t> {
public:
  MaxScalarStorage()
    : last_value(0), final_value(0)
  {}

  ~MaxScalarStorage()
  {}

  virtual void reinit()
  {
    last_value = 0;
    final_value = 0;
  }

public:
  // modifier
  virtual void set(uint64_t val)
  {
    if (val > final_value) {
      final_value = val;
    }

    last_value = val;
  }

  virtual void inc(uint64_t val) {
    last_value += val;
    if (last_value > final_value) {
      final_value = last_value;
    }
  }

  virtual void dec(uint64_t val) {
    assert(last_value >= val);
    last_value -= val;
  }

public:
  // accessor
  virtual void finish()           {}
  virtual uint64_t result() const { return final_value; }

private:
  uint64_t last_value;
  uint64_t final_value;
};

// storage class for Scalar that counts
class CountScalarStorage : ScalarStorage<uint64_t> {
public:
  CountScalarStorage()
    : final_value(0)
  {}

  ~CountScalarStorage()
  {}

  virtual void reinit()
  {
    final_value = 0;
  }

public:
  // modifier
  virtual void set(uint64_t val)
  {
    final_value = val;
  }

  virtual void inc(uint64_t val) {
    final_value += val;
  }

  virtual void dec(uint64_t val) {
    assert(final_value >= val);
    final_value -= val;
  }

public:
  // accessor
  virtual void finish()           {}
  virtual uint64_t result() const { return final_value; }

private:
  uint64_t final_value;
};

// storage class for Scalar that averages over cycle
class AverageScalarStorage : ScalarStorage<double> {
public:
  AverageScalarStorage()
    : last_value(0), last_cycle(0), total_value(0)
  {}

  ~AverageScalarStorage()
  {}

  virtual void reinit()
  {
    last_value = 0;
    last_cycle = get_curr_cycle();
    total_value = 0;
  }

public:
  // modifier
  virtual void set(uint64_t val)
  {
    total_value += last_value * (get_curr_cycle() - last_cycle);
    last_cycle = get_curr_cycle();
    last_value = val;
  }

  virtual void inc(uint64_t val) {
    set(last_value + val);
  }

  virtual void dec(uint64_t val) {
    assert(last_value >= val);
    set(last_value - val);
  }

public:
  // accessor
  virtual void finish()         { set(last_value); }
  virtual double result() const { return double(total_value) / double(last_cycle); }

private:
  uint64_t last_value;
  uint64_t last_cycle;

  uint64_t total_value;
};

// template class for statistics that results in single value
template<typename Storage, typename T>
class ScalarBase : public StatBase {
public:
  ScalarBase()   {}
  ~ScalarBase()  {}

  void operator++()     { storage.inc(1); }
  void operator--()     { storage.dec(1); }
  void operator++(int)  { ++*this;        }
  void operator--(int)  { --*this;        }

  void operator=(uint64_t rhs)  { storage.set(rhs); }
  void operator+=(uint64_t rhs) { storage.inc(rhs); }
  void operator-=(uint64_t rhs) { storage.dec(rhs); }

  void finalize()       { storage.finish(); }
  T get_result() const  { return storage.result(); }

  void reinit()         { storage.reinit(); }

private:
  Storage storage;
};

// Statistics for maximum value
class Maximum : public ScalarBase<MaxScalarStorage, uint64_t> {
  public:
    using ScalarBase<MaxScalarStorage, uint64_t>::operator=;
};

// Statistics for counting
class Counter : public ScalarBase<CountScalarStorage, uint64_t> {
  public:
    using ScalarBase<CountScalarStorage, uint64_t>::operator=;
};

// Statistics for counting
class Average : public ScalarBase<AverageScalarStorage, double> {
  public:
    using ScalarBase<AverageScalarStorage, double>::operator=;
};

} // namespace Stats

#endif // __STATS_H__

