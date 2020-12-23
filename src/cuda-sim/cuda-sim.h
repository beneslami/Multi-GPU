// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef CUDASIM_H_INCLUDED
#define CUDASIM_H_INCLUDED

#include "../abstract_hardware_model.h"
#include"../gpgpu-sim/shader.h"
#include <stdlib.h>
#include <map>
#include <string>
#include"ptx_sim.h"

class memory_space;
class function_info;
class symbol_table;

extern const char *g_gpgpusim_version_string;
extern int g_ptx_sim_mode;
extern int g_debug_execution;
extern int g_debug_thread_uid;
extern void ** g_inst_classification_stat;
extern void ** g_inst_op_classification_stat;
extern int g_ptx_kernel_count; // used for classification stat collection purposes 

void ptx_opcocde_latency_options (option_parser_t opp);
extern class kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                            gpgpu_ptx_sim_arg_list_t args, 
                                            struct dim3 gridDim, 
                                            struct dim3 blockDim, 
                                                          class gpgpu_t *gpu );
extern void gpgpu_cuda_ptx_sim_main_func( kernel_info_t &kernel, bool openCL = false );
extern void   print_splash();
extern void   gpgpu_ptx_sim_register_const_variable(void*, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, gpgpu_t *gpu );

extern void read_sim_environment_variables();
extern void ptxinfo_opencl_addinfo( std::map<std::string,function_info*> &kernels );
const warp_inst_t *ptx_fetch_inst( address_type pc );
const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info(const class function_info *kernel);
void ptx_print_insn( address_type pc, FILE *fp );
std::string ptx_get_insn_str( address_type pc );
void set_param_gpgpu_num_shaders(int num_shaders);

class warp_context_t;

struct barrier_context_t {
  warp_set_t m_warps_mapped;
  warp_set_t m_warps_active;
  warp_set_t m_warps_at_barrier;
};

/*
 * Context switching
 */
class SimulationInitializer {
private:
  // shared_memory_lookup[pid][sm_idx] = shared memory
  static std::map<unsigned, std::map<unsigned,memory_space*> > shared_memory_lookup;
  // ptx_cta_lookup[pid][sm_idx] = shared memory
  static std::map<unsigned, std::map<unsigned,ptx_cta_info*> > ptx_cta_lookup;
  // local_memory_lookup[pid][sid][tid] = shared memory
  static std::map<unsigned, std::map<unsigned, std::map<unsigned,memory_space*> > > local_memory_lookup;

  // shared_memory_saved[pid][flat_cta_id] = saved context
  static std::map<unsigned, std::map<unsigned, memory_space*> > shared_memory_saved;
  // ptx_cta_info_saved[pid][flat_cta_id] = saved ptx_cta_info
  static std::map<unsigned, std::map<unsigned, ptx_cta_info*> > ptx_cta_saved;
  // local_memory_saved[pid][flat_cta_id][flat_thread_id] = saved local memory
  static std::map<unsigned, std::map<unsigned, std::map<unsigned, memory_space*> > > local_memory_saved;
  // thread_info_saved[pid][flat_cta_id][flat_thread_id] = saved ptx_thread_info
  static std::map<unsigned, std::map<unsigned, std::map<unsigned, ptx_thread_info*> > > thread_info_saved;
  // warp_saved[pid][flat_cta_id][wid] = saved shd_warp_t
  static std::map<unsigned, std::map<unsigned, std::map<unsigned, warp_context_t*> > > warp_context_saved;
  // simt_stack_saved[pid][flat_cta_id][wid] = saved simt_stack
  static std::map<unsigned, std::map<unsigned, std::map<unsigned, simt_stack*> > > simt_stack_saved;
  // barrier_context_saved[pid][flat_cta_id] = saved barrier_context_t
  static std::map<unsigned, std::map<unsigned, barrier_context_t> > barrier_context_saved;
  // cta_stat_saved[pid][flat_cta_id] = saved cta_stat_context_t
  static std::map<unsigned, std::map<unsigned, cta_stat_context_t> > cta_stat_saved;

public:
  static unsigned ptx_sim_init_thread( kernel_info_t &kernel,
                                       class ptx_thread_info** thread_info,
                                       int sid,
                                       unsigned tid,
                                       unsigned threads_left,
                                       unsigned num_threads, 
                                       class core_t *core, 
                                       unsigned hw_cta_id, 
                                       unsigned hw_warp_id,
                                       gpgpu_t *gpu,
                                       bool& context_loading,
                                       bool functionalSimulationMode = false);

  static void save_cta_context( kernel_info_t* kernel,
                                unsigned sid,
                                unsigned flat_cta_id,
                                unsigned hw_cta_id);

  static void save_thread_context( kernel_info_t* kernel,
                                   unsigned sid,
                                   unsigned flat_cta_id,
                                   unsigned hw_tid_begin,
                                   ptx_thread_info* threadInfo);

  static void save_warp_context( kernel_info_t* kernel,
                                 unsigned flat_cta_id,
                                 unsigned logical_wid,
                                 const shd_warp_t& warp,
                                 simt_stack* stackInfo);

  static void save_barrier_context( kernel_info_t* kernel,
                                    unsigned flat_cta_id,
                                    unsigned hw_wid_begin,
                                    warp_set_t warps_mapped,
                                    warp_set_t warps_active,
                                    warp_set_t warps_at_barrier);

private:
  static ptx_cta_info* load_context(kernel_info_t& kernel,
                                    unsigned sid,
                                    unsigned tid,
                                    unsigned lookup_idx,
                                    std::list<ptx_thread_info *>& active_threads);

  static void load_warp_context(kernel_info_t& kernel,
                                unsigned hw_cta_id,
                                unsigned hw_wid_start,
                                shader_core_ctx* core);

  static void load_barrier_context( kernel_info_t& kernel,
                                    unsigned hw_cta_id,
                                    unsigned hw_wid_start,
                                    shader_core_ctx* core);

  static void load_cta_stat_context(kernel_info_t& kernel,
                                    unsigned sid,
                                    unsigned hw_cta_id);
};

/*!
 * This class functionally executes a kernel. It uses the basic data structures and procedures in core_t 
 */
class functionalCoreSim: public core_t
{    
public:
    functionalCoreSim(kernel_info_t * kernel, gpgpu_sim *g, unsigned warp_size)
        : core_t( g, kernel, warp_size, kernel->threads_per_cta() )
    {
        m_warpAtBarrier =  new bool [m_warp_count];
        m_liveThreadCount = new unsigned [m_warp_count];
    }
    virtual ~functionalCoreSim(){
        warp_exit(0);
        delete[] m_liveThreadCount;
        delete[] m_warpAtBarrier;
    }
    //! executes all warps till completion 
    void execute();
    virtual void warp_exit( unsigned warp_id );
    virtual bool warp_waiting_at_barrier( unsigned warp_id ) const  
    {
        return (m_warpAtBarrier[warp_id] || !(m_liveThreadCount[warp_id]>0));
    }
    
private:
    void executeWarp(unsigned, bool &, bool &);
    //initializes threads in the CTA block which we are executing
    void initializeCTA();
    virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)
    {
    if(m_thread[tid]==NULL || m_thread[tid]->is_done()){
        m_liveThreadCount[tid/m_warp_size]--;
        }
    }
    
    // lunches the stack and set the threads count
    void  createWarp(unsigned warpId);
    
    //each warp live thread count and barrier indicator
    unsigned * m_liveThreadCount;
    bool* m_warpAtBarrier;
};

#define RECONVERGE_RETURN_PC ((address_type)-2)
#define NO_BRANCH_DIVERGENCE ((address_type)-1)
address_type get_return_pc( void *thd );
const char *get_ptxinfo_kname();
void print_ptxinfo();
void clear_ptxinfo();
struct gpgpu_ptx_sim_kernel_info get_ptxinfo_kinfo();

#endif
