#ifndef __NAMED_INFO_H__
#define __NAMED_INFO_H__

#include <map>
#include <string>
#include <vector>
#include <stdint.h>

template<typename T>
class NamedInfo {
  public:
    NamedInfo() {}
    ~NamedInfo() {}

    // add update info
    void add(std::string str, T value);
    // does it have update info?
    bool has(std::string str) const;
    // get the update info
    T get(std::string str) const;

  private:
    std::map<std::string, T> values;
};

typedef NamedInfo<uint64_t> SchedulerUpdateInfo;
typedef NamedInfo<std::vector<std::string> > SchedulerInfo;

#endif // __NAMED_INFO_H__

