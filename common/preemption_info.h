#ifndef PREEMPTION_INFO_H_
#define PREEMPTION_INFO_H_

#include <map>
#include <cassert>

class shader_preempt_overhead;

class preemption_info_t {
public:
    preemption_info_t(unsigned sid);
    preemption_info_t(const shader_preempt_overhead& item);

    unsigned get_sid() const { return m_sid; }
    unsigned get_preempting_ctas() const { return m_techniques.size(); }

    bool is_cta_draining(unsigned hw_cta_id) const
    {
      assert(m_techniques.find(hw_cta_id) != m_techniques.end());
      return m_techniques.at(hw_cta_id) == PREEMPT_WITH_DRAINING;
    }
    bool is_cta_switching(unsigned hw_cta_id) const
    {
      assert(m_techniques.find(hw_cta_id) != m_techniques.end());
      return m_techniques.at(hw_cta_id) == PREEMPT_WITH_SWITCHING;
    }
    bool is_cta_flushing(unsigned hw_cta_id) const
    {
      assert(m_techniques.find(hw_cta_id) != m_techniques.end());
      return m_techniques.at(hw_cta_id) == PREEMPT_WITH_FLUSHING;
    }

    void make_cta_drain(unsigned hw_cta_id)
    {
      assert(m_techniques.find(hw_cta_id) == m_techniques.end());
      m_techniques[hw_cta_id] = PREEMPT_WITH_DRAINING;
    }
    void make_cta_switch(unsigned hw_cta_id)
    {
      assert(m_techniques.find(hw_cta_id) == m_techniques.end());
      m_techniques[hw_cta_id] = PREEMPT_WITH_SWITCHING;
    }
    void make_cta_flush(unsigned hw_cta_id)
    {
      assert(m_techniques.find(hw_cta_id) == m_techniques.end());
      m_techniques[hw_cta_id] = PREEMPT_WITH_FLUSHING;
    }

private:
    enum PREEMPTION_TECHNIQUE_ENUM {
      PREEMPT_WITH_DRAINING,
      PREEMPT_WITH_SWITCHING,
      PREEMPT_WITH_FLUSHING,
      PREEMPTION_TECHNIQUE_NUM,
    };

    unsigned m_sid; // shader id
    // m_techniques[hw_cta_id] = PREEMPTION_TECHNIQUE
    std::map<unsigned, PREEMPTION_TECHNIQUE_ENUM> m_techniques;

public:
    typedef std::map<unsigned, PREEMPTION_TECHNIQUE_ENUM>::iterator       iterator;
    typedef std::map<unsigned, PREEMPTION_TECHNIQUE_ENUM>::const_iterator const_iterator;

    iterator        begin()       { return m_techniques.begin(); }
    iterator        end()         { return m_techniques.end();   }
    const_iterator  begin() const { return m_techniques.begin(); }
    const_iterator  end()   const { return m_techniques.end();   }
};

#endif // PREEMPTION_INFO_H_

