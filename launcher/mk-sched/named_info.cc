#include <cassert>

#include "named_info.h"

template<typename T>
void
NamedInfo<T>::add(std::string str, T value)
{
  assert(values.find(str) == values.end());
  values[str] = value;
}

template<typename T>
bool
NamedInfo<T>::has(std::string str) const
{
  return values.find(str) != values.end();
}

template<typename T>
T
NamedInfo<T>::get(std::string str) const
{
  typename std::map<std::string, T>::const_iterator it = values.find(str);
  assert(it != values.end());
  return it->second;
}

template class NamedInfo<uint64_t>;
template class NamedInfo<std::vector<std::string> >;

