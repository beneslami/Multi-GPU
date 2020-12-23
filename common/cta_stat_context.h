#ifndef CTA_STAT_CONTEXT_H_
#define CTA_STAT_CONTEXT_H_

class cta_stat_context_t {
public:
    // constructor
    cta_stat_context_t();

    // accessor
    bool is_valid() const { return m_valid; }
    unsigned long long get_executed_insts() const;
    unsigned long long get_executed_cycles() const;
    unsigned long long get_started_cycle() const;

    // modifier
    void start_stat();
    void inc_executed_insts() { ++m_executed_insts; }

    // helper method
    void save_context(const cta_stat_context_t& to_save);
    void load_context(const cta_stat_context_t& to_load);
    void clear_context();

public:
    bool m_valid;
    bool m_saved;
    unsigned long long m_executed_insts;
    union {
      unsigned long long m_started_cycles;
      unsigned long long m_executed_cycles;
    };
};

#endif // CTA_STAT_CONTEXT_H_

