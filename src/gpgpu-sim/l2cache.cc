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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <set>

#include "../option_parser.h"
#include "mem_fetch.h"
#include "dram.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"
#include "shader.h"
#include "mem_latency_stat.h"
#include "l2cache_trace.h"
//#include "../ramulator_sim/Config.h"
#include <sstream>
extern unsigned long long rop_in;
extern unsigned long long rop_out;
extern unsigned long long icnt_L2_in;
extern unsigned long long icnt_L2_out;
extern unsigned long long L2_dram_in;
extern unsigned long long L2_dram_out;
extern unsigned long long dram_latency_in;
extern unsigned long long dram_latency_out;
extern unsigned long long dram_L2_in;
extern unsigned long long dram_L2_out;
extern unsigned long long returnq_out;
extern unsigned long long returnq_out_delete;
extern unsigned long long returnq_out_local;
extern unsigned long long returnq_out_inter;
extern unsigned long long returnq_out_inter_pop;
extern unsigned long long returnq_out_inter_pop_delete;

mem_fetch * partition_mf_allocator::alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const 
{
    assert( wr );
    mem_access_t access( type, addr, size, wr );
    mem_fetch *mf = new mem_fetch( access, 
                                   NULL,
                                   WRITE_PACKET_SIZE, 
                                   -1, 
                                   -1, 
                                   -1,
                                   m_memory_config );
    return mf;
}

new_addr_type *kain_cache[4];
static int kain_init = 0;
memory_partition_unit::memory_partition_unit( unsigned partition_id, 
                                              const struct memory_config *config,
                                              class memory_stats_t *stats )
: m_id(partition_id), m_config(config), m_stats(stats), m_arbitration_metadata(config) 
{
    m_dram = new dram_t(m_id,m_config,m_stats,this);
//	Config m_r_config("HBM-config.cfg");
//	m_r_config.set_core_num(core_numbers);

//	m_dram_r = new GpuWrapper(m_r_config, m_config->m_L2_config.get_line_sz() , this, m_id);

    if(kain_init == 0)
    {
        kain_init = 1;
        for(int j = 0; j < 4; j++)
        {
            kain_cache[j] = new new_addr_type[8388608];//1024*1024
                for(unsigned i = 0; i < 8388608; i++)
                    kain_cache[j][i] = 0;
        }
    }

    m_sub_partition = new memory_sub_partition*[m_config->m_n_sub_partition_per_memory_channel]; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned sub_partition_id = m_id * m_config->m_n_sub_partition_per_memory_channel + p; 
        m_sub_partition[p] = new memory_sub_partition(sub_partition_id, m_config, stats); 
    }
}

memory_partition_unit::~memory_partition_unit() 
{
    delete m_dram; 
//	delete m_dram_r;
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        delete m_sub_partition[p]; 
    } 
    delete[] m_sub_partition; 
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(const struct memory_config *config) 
: m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1), 
  m_private_credit(config->m_n_sub_partition_per_memory_channel, 0), 
  m_shared_credit(0) 
{
    // each sub partition get at least 1 credit for forward progress 
    // the rest is shared among with other partitions 
    m_private_credit_limit = 1; 
    m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size 
                            + config->gpgpu_dram_return_queue_size 
                            - (config->m_n_sub_partition_per_memory_channel - 1); 
    if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 
        or config->gpgpu_dram_return_queue_size == 0) 
    {
        m_shared_credit_limit = 0; // no limit if either of the queue has no limit in size 
    }
    assert(m_shared_credit_limit >= 0); 
}

bool memory_partition_unit::arbitration_metadata::has_credits(int inner_sub_partition_id) const 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] < m_private_credit_limit) {
        return true; 
    } else if (m_shared_credit_limit == 0 || m_shared_credit < m_shared_credit_limit) {
        return true; 
    } else {
        return false; 
    }
}

//ZSQ //
void memory_partition_unit::arbitration_metadata::borrow_credit(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] < m_private_credit_limit) {
        m_private_credit[spid] += 1; 
    } else //if (m_shared_credit_limit == 0 || m_shared_credit < m_shared_credit_limit) {
        m_shared_credit += 1; 
//    } else {
//        assert(0 && "DRAM arbitration error: Borrowing from depleted credit!"); 
//    }
    m_last_borrower = spid; 
}

void memory_partition_unit::arbitration_metadata::return_credit(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] > 0) {
        m_private_credit[spid] -= 1; 
    } else {
        m_shared_credit -= 1; 
    } 
    assert((m_shared_credit >= 0) && "DRAM arbitration error: Returning more than available credits!"); 
}

void memory_partition_unit::arbitration_metadata::print( FILE *fp ) const 
{
    fprintf(fp, "private_credit = "); 
    for (unsigned p = 0; p < m_private_credit.size(); p++) {
        fprintf(fp, "%d ", m_private_credit[p]); 
    }
    fprintf(fp, "(limit = %d)\n", m_private_credit_limit); 
    fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit, m_shared_credit_limit); 
}

bool memory_partition_unit::busy() const 
{
    bool busy = false; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        if (m_sub_partition[p]->busy()) {
            busy = true; 
        }
    }
    return busy; 
}

void memory_partition_unit::cache_cycle(unsigned cycle) 
{
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->cache_cycle(cycle); 
    }
}

void memory_partition_unit::visualizer_print( gzFile visualizer_file ) const 
{
/*
    m_dram->visualizer_print(visualizer_file);
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->visualizer_print(visualizer_file); 
    }
*/
}

// determine whether a given subpartition can issue to DRAM 
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full(); 
    bool has_dram_resource = m_arbitration_metadata.has_credits(spid); 

    MEMPART_DPRINTF("sub partition %d sub_partition_contention=%c has_dram_resource=%c\n", 
                    spid, (sub_partition_contention)? 'T':'F', (has_dram_resource)? 'T':'F'); 

    return (has_dram_resource && !sub_partition_contention); 
}

#if SM_SIDE_LLC == 1
   extern unsigned long long gpu_sim_cycle;
   extern unsigned long long gpu_tot_sim_cycle;
   bool memory_partition_unit::dram_latency_avaliable() {
        for (int i = 0; i < m_config->m_n_sub_partition_per_memory_channel; i++) {
            if (can_issue_to_dram(i)) return true;
        }
        return false;
   }
   void memory_partition_unit::receive_inter_icnt(mem_fetch *mf){
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency+32;
                m_dram_latency_queue.push_back(d);
	    	dram_latency_in++;
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
   }
#endif

int memory_partition_unit::global_sub_partition_id_to_local_id(int global_sub_partition_id) const
{
    return (global_sub_partition_id - m_id * m_config->m_n_sub_partition_per_memory_channel); 
}


class KAIN_GPU_chiplet KAIN_NoC_r(3);
long long KAIN_HBM_Cache_hit = 0;
long long KAIN_HBM_Cache_miss = 0;

int kain_memory_page_count[4];
extern std::map<new_addr_type, new_addr_type*> KAIN_page_table;
#if HBM_CACHE == 1
std::list<mem_fetch*> KAIN_HBM_Cache_request[32];
#endif
void memory_partition_unit::dram_cycle() 
{ 

//ZSQ 20210130 Rearranged the above piece of code here
#if SM_SIDE_LLC == 1
    unsigned _mid = m_id;
    unsigned _subid = _mid * 2;
    std::ostringstream out;
    if (!KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid)) {   //returnq turn, start
        mem_fetch *mf_return = m_dram->return_queue_top();

        if (mf_return) {    //returnq turn, m_dram_r->r_return_queue_top() != NULL, start
            if (mf_return->get_sid() / 32 != mf_return->get_chip_id() / 8) { //remote, push to inter_icnt
                unsigned to_module = 192 + mf_return->get_sid() / 32;
                unsigned from_module = 192 + mf_return->get_chip_id() / 8;
                if (INTER_TOPO == 1 &&
                    (mf_return->get_sid() / 32 + mf_return->get_chip_id() / 8) % 2 == 0) { //ring, forward
                    to_module = 192 + (mf_return->get_sid() / 32 + 1) % 4;
                    mf_return->set_next_hop(to_module);
                }
                unsigned response_size = mf_return->get_is_write() ? mf_return->get_ctrl_size() : mf_return->size();
                if (::icnt_has_buffer(from_module, response_size)) {
                    ::icnt_push(from_module, to_module, (void *) mf_return, response_size);
                    m_dram->return_queue_pop();
                    returnq_out++;
                    returnq_out_inter++;
                    if(gpu_sim_cycle > 100) {
                        out << "DRAM_icnt\tsrc: " << mf_return->get_src() << "\tdst: " << mf_return->get_dst() <<
                            "\tID: " << mf_return->get_request_uid() << "\ttype: " << mf_return->get_type()
                            << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf_return->get_sid() / 32 << "\tsize: "
                            << response_size << "\tgpu_cycle: " << gpu_sim_cycle << "\n";
                        std::fstream outdata;
                        outdata.open("report.txt", std::ios_base::app);
                        outdata << out.str().c_str();
                        outdata.close();
                    }
                }
            }
            else { //local, push to dram_L2_queue
                unsigned dest_global_spid = mf_return->get_sub_partition_id();
                int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
                if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                    if (mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                        returnq_out_delete++;
                        printf("ZSQ: should not arrive here.\n");
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                        m_arbitration_metadata.return_credit(dest_spid);
                        MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return,
                                        dest_spid);
                    }
                    m_dram->return_queue_pop();
                    returnq_out++;
                    returnq_out_local++;
//printf("		local, WRBK delete or ->dram_L2_queue, dest_global_spid = %d, dest_spid = %d\n", dest_global_spid, dest_spid);
                }
            }
            KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid);
        }   //returnq turn, m_dram_r->r_return_queue_top() != NULL, end
        else {    //returnq turn, m_dram_r->r_return_queue_top() = NULL, start
            m_dram->return_queue_pop();
//printf("        NULL, returnq size = %d\n", m_dram_r->r_returnq_size());
            if (!m_sub_partition[0]->dram_L2_queue_full() && !m_sub_partition[1]->dram_L2_queue_full()) {
                if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid + 1)) {
                    if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid + 1)) {
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid + 1);
                        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid + 1);
                    } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
                } else {
                    if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
                        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid + 1);
                    } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid + 1))
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid + 1);
                }
                if (mf_return) {
                    unsigned dest_global_spid = mf_return->get_sub_partition_id();
                    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                    m_arbitration_metadata.return_credit(dest_spid);
                    if (mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                        returnq_out_delete++;
                        returnq_out_inter_pop_delete++;
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                        returnq_out_inter_pop++;
                    }
                }
            }

/*	mf_return = NULL;
        int pop_flag = -1;
        if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid+1)) {
            if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1))
                mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
            if (mf_return) {
                KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
                pop_flag = 1;
            } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
                mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
                if (mf_return) pop_flag = 0;
            }
        } else {
            if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
                mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
            if (mf_return) {
                KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
                pop_flag = 0;
            } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) {
                mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
                if (mf_return) pop_flag = 1;
            }
        }
        if (mf_return) {
            unsigned dest_global_spid = mf_return->get_sub_partition_id();
            int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
            if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                m_arbitration_metadata.return_credit(dest_spid); 
                if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) { 
                    m_sub_partition[dest_spid]->set_done(mf_return);
                    delete mf_return;
		    returnq_out_delete++;
		    returnq_out_inter_pop_delete++;
//printf("		icnt_pop_llc[%d] pop, mf sid = %d, chip_id = %d, sub_id = %d, WRBK_ACC, delete, dest_spid = %d\n", pop_flag, mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id(), dest_spid);
                } else {
                    m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                    mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
		    returnq_out_inter_pop++;
//printf("		icnt_pop_llc[%d] pop, mf sid = %d, chip_id = %d, sub_id = %d, !WRBK_ACC, ->dram_L2_queue, dest_spid = %d\n", pop_flag, mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id(), dest_spid);
                }
                if (pop_flag == 0)
                    KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
                else if (pop_flag == 1)
                    KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
            }
        } else {
            if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
            if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
        }
*/    }   //returnq turn, m_dram_r->r_return_queue_top() = NULL, end
    } //returnq turn, end
    else { // inter_icnt_pop_llc turn, start
        mem_fetch *mf_return = NULL;
        bool flag = false;
        if (!m_sub_partition[0]->dram_L2_queue_full() && !m_sub_partition[1]->dram_L2_queue_full()) {
            if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid + 1)) {
                if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid + 1)) {
                    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid + 1);
                    KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid + 1);
                } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
                    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
            } else {
                if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
                    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
                    KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid + 1);
                } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid + 1))
                    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid + 1);
            }
            if (mf_return) {
                flag = true;
                unsigned dest_global_spid = mf_return->get_sub_partition_id();
                int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                m_arbitration_metadata.return_credit(dest_spid);
                if (mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                    m_sub_partition[dest_spid]->set_done(mf_return);
                    delete mf_return;
                    returnq_out_delete++;
                    returnq_out_inter_pop_delete++;
                } else {
                    m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                    mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                    returnq_out_inter_pop++;
                }
            }
        }
/*    int pop_flag = -1;
    if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid+1)) {
        if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1))
             mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
        if (mf_return) {
            KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
            pop_flag = 1;
        } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
            mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
            if (mf_return) pop_flag = 0;
        }
    } else {
        if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
             mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
        if (mf_return) {
            KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
            pop_flag = 0;
        } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) {
            mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
            if (mf_return) pop_flag = 1;
        }
    }
    if (mf_return) {    //inter_icnt_pop_llc turn, top() != NULL, start
        unsigned dest_global_spid = mf_return->get_sub_partition_id();
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            m_arbitration_metadata.return_credit(dest_spid);
            if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                m_sub_partition[dest_spid]->set_done(mf_return);
                delete mf_return;
		returnq_out_delete++;
		returnq_out_inter_pop_delete++;
//printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle(), ->dram_L2_queue, inter_llc turn, icnt_pop_llc[%d] pop, mf sid = %d, chip_id = %d, sub_id = %d, WRBK_ACC, delete, dest_spid = %d\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id, pop_flag, mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id(), dest_spid);
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
		returnq_out_inter_pop++;
//printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle(), ->dram_L2_queue, inter_llc turn, icnt_pop_llc[%d] pop, mf sid = %d, chip_id = %d, sub_id = %d, !WRBK_ACC, ->dram_L2_queue, dest_spid = %d\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id, pop_flag, mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id(), dest_spid);
            }
        }
        if (pop_flag == 0)
            KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
        else if (pop_flag == 1)
            KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid);
    }   //inter_icnt_pop_llc turn, top() != NULL, end
    else {    //inter_icnt_pop_llc turn, top() = NULL, start
//        if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
//        if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
*/
        if (!flag) {
            mf_return = m_dram->return_queue_top();
//printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle(), ->dram_L2_queue, inter_llc turn, r_return_queue_top()\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id);
            if (mf_return) {
//printf("	!NULL, mf sid = %d, chip_id = %d, sub_id = %d\n", mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id());
                if (mf_return->get_sid() / 32 != mf_return->get_chip_id() / 8) { //remote, push to inter_icnt
                    unsigned to_module = 192 + mf_return->get_sid() / 32;
                    unsigned from_module = 192 + mf_return->get_chip_id() / 8;
                    mf_return->set_src(from_module);
                    mf_return->set_dst(to_module);
                    mf_return->set_next_hop(to_module);
                    if (INTER_TOPO == 1 &&
                        (mf_return->get_sid() / 32 + mf_return->get_chip_id() / 8) % 2 == 0) //ring, forward
                        to_module = 192 + (mf_return->get_sid() / 32 + 1) % 4;
                    unsigned response_size = mf_return->get_is_write() ? mf_return->get_ctrl_size() : mf_return->size();
                    if (::icnt_has_buffer(from_module, response_size)) {
                        ::icnt_push(from_module, to_module, (void *) mf_return, response_size);
                        m_dram->return_queue_pop();
                        returnq_out++;
                        returnq_out_inter++;
                        if(gpu_sim_cycle > 100) {
                            out << "DRAM_icnt\tsrc: " << mf_return->get_src() << "\tdst: " << mf_return->get_dst() <<
                                "\tID: " << mf_return->get_request_uid() << "\ttype: " << mf_return->get_type()
                                << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf_return->get_sid() / 32 << "\tsize: "
                                << response_size << "\tgpu_cycle: " << gpu_sim_cycle << "\n";
                            std::fstream outdata;
                            outdata.open("report.txt", std::ios_base::app);
                            outdata << out.str().c_str();
                            outdata.close();
                        }
                    }
                } else { //local, push to dram_L2_queue
                    unsigned dest_global_spid = mf_return->get_sub_partition_id();
                    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
                    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                        if (mf_return->get_access_type() == L1_WRBK_ACC ||
                            mf_return->get_access_type() == L2_WRBK_ACC) {
                            m_sub_partition[dest_spid]->set_done(mf_return);
                            delete mf_return;
                            returnq_out_delete++;
                            printf("ZSQ: should not arrive here.\n");
                        } else {
                            m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                            mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                            m_arbitration_metadata.return_credit(dest_spid);
                            MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return,
                                            dest_spid);
                        }
                        m_dram->return_queue_pop();
                        returnq_out++;
                        returnq_out_local++;
                    }
                }
            } else {
                m_dram->return_queue_pop();
//printf("	NULL, returnq size = %d\n", m_dram_r->r_returnq_size());
            }
        } //inter_icnt_pop_llc turn, top() = NULL, end
    }   //inter_icnt_pop_llc turn, end
#endif
//ZSQ 20210130 Rearranged the above piece of code here

         
#if SM_SIDE_LLC == 0
    mem_fetch* mf_return = m_dram->return_queue_top();
    if (mf_return) {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
        if (!m_sub_partition[0]->dram_L2_queue_full()&&!m_sub_partition[1]->dram_L2_queue_full()) {
            if( mf_return->get_access_type() == L1_WRBK_ACC ) {
                m_sub_partition[dest_spid]->set_done(mf_return);
                delete mf_return;
        returnq_out_delete++;
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.return_credit(dest_spid);
                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
            }
            m_dram->return_queue_pop();
               returnq_out++;
        }
    } else {
        m_dram->return_queue_pop();
    }
#endif

        m_dram->cycle(); // In this part, when read/write complete, the return q should be automatically written due to the call back function.

//ZSQ 20210130 Rearranged the above piece of code here
#if SM_SIDE_LLC == 1
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) { //loop sub_partition start
        int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;
        if(!KAIN_NoC_r.get_inter_icnt_pop_mem_turn(m_id)){  //L2_dram_queue turn, start
            if (!m_sub_partition[spid]->L2_dram_queue_empty()) {    //L2_dram_queue turn, !L2_dram_queue_empty, start
                mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
                if (mf->get_sid()/32 != mf->get_chip_id()/8){ //remote, push to inter_icnt
                    unsigned from_module = 192 + mf->get_sid()/32;
                    unsigned to_module = 192 + mf->get_chip_id()/8;
                    mf->set_src(from_module);
                    mf->set_dst(to_module);
                    mf->set_next_hop(to_module);
                    if (INTER_TOPO == 1 && (mf->get_sid()/32+mf->get_chip_id()/8)%2 == 0) { //ring, forward
                        to_module = 192 + (mf->get_chip_id() / 8 + 1) % 4;
                        mf->set_next_hop(to_module);
                    }
                    unsigned size = mf->get_is_write()?mf->size():mf->get_ctrl_size();
                    if (::icnt_has_buffer(from_module, size && m_arbitration_metadata.has_credits(spid))) {
                        ::icnt_push(from_module, to_module, (void*)mf, size);
                        m_sub_partition[spid]->L2_dram_queue_pop();
                        m_arbitration_metadata.borrow_credit(spid);
                    } //else printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle() 4, L2_dram_queue_top() but !icnt_has_buffer(%d)\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id, from_module);
                } else if (can_issue_to_dram(spid) && m_arbitration_metadata.has_credits(spid)){ //local. push to dram_latency_queue
                    m_sub_partition[spid]->L2_dram_queue_pop();
                    MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                    dram_delay_t d;
                    d.req = mf;
                    d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                    m_dram_latency_queue.push_back(d);
	    	    dram_latency_in++;
                    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                    m_arbitration_metadata.borrow_credit(spid);
                    break;  // the DRAM should only accept one request per cycle
                }
                KAIN_NoC_r.set_inter_icnt_pop_mem_turn(m_id);
            }    //L2_dram_queue turn, !L2_dram_queue_empty, end
            else {   //L2_dram_queue turn, L2_dram_queue_empty, start
                if (!KAIN_NoC_r.inter_icnt_pop_mem_empty(m_id)) {
                    mem_fetch *mf =  KAIN_NoC_r.inter_icnt_pop_mem_pop(m_id);
                    if(gpu_sim_cycle >= 100 && mf) {
                        out1 << "icnt_mem_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                             "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                             << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_sid()/32 << "\tsize: " << mf->size()
                             <<"\tgpu_cycle: " << gpu_sim_cycle << "\n";
                        rep->apply(out1.str().c_str());
                    }
                    dram_delay_t d;
                    d.req = mf;
                    d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                    m_dram_latency_queue.push_back(d);
	    	    dram_latency_in++;
                    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                }
            }    //L2_dram_queue turn, L2_dram_queue_empty, end
        }   //L2_dram_queue turn, end
        else {  //inter_icnt_pop_mem, start
            if (!KAIN_NoC_r.inter_icnt_pop_mem_empty(m_id)) {   //inter_icnt_pop_mem turn, !inter_icnt_pop_mem_empty, start
                mem_fetch *mf =  KAIN_NoC_r.inter_icnt_pop_mem_pop(m_id);
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                m_dram_latency_queue.push_back(d);
	    	dram_latency_in++;
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                KAIN_NoC_r.set_inter_icnt_pop_mem_turn(m_id);
            }   //inter_icnt_pop_mem turn, !inter_icnt_pop_mem_empty, end
            else {  //inter_icnt_pop_mem turn, inter_icnt_pop_mem_empty, start
                if (!m_sub_partition[spid]->L2_dram_queue_empty()) {
                    mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
                    if (mf->get_sid()/32 != mf->get_chip_id()/8){ //remote, push to inter_icnt
                        unsigned from_module = 192 + mf->get_sid()/32;
                        unsigned to_module = 192 + mf->get_chip_id()/8;
                        if (INTER_TOPO == 1 && (mf->get_sid()/32+mf->get_chip_id()/8)%2 == 0) //ring, forward
                            to_module = 192 + (mf->get_chip_id()/8+1)%4;
                        unsigned size = mf->get_is_write()?mf->size():mf->get_ctrl_size();
                        if (::icnt_has_buffer(from_module, size) && m_arbitration_metadata.has_credits(spid)) {
                            ::icnt_push(from_module, to_module, (void*)mf, size);
                            m_sub_partition[spid]->L2_dram_queue_pop();
                            m_arbitration_metadata.borrow_credit(spid);
                        } //else printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle() 4, L2_dram_queue_top() but !icnt_has_buffer(%d)\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id, from_module);
                    }
                    else if (can_issue_to_dram(spid) && m_arbitration_metadata.has_credits(spid)){ //local. push to dram_latency_queue
                        m_sub_partition[spid]->L2_dram_queue_pop();
                        MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                        dram_delay_t d;
                        d.req = mf;
                        d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                        m_dram_latency_queue.push_back(d);
	    	        dram_latency_in++;
                        mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                        m_arbitration_metadata.borrow_credit(spid);
                        break;  // the DRAM should only accept one request per cycle
                    }
                }
            }   //inter_icnt_pop_mem turn, inter_icnt_pop_mem_empty, end
        }   //inter_icnt_pop_mem, end
    }   //loop partiton end
#endif
//ZSQ 20210130 Rearranged the above piece of code here

#if SM_SIDE_LLC == 0
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
            int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;
            if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid) && m_arbitration_metadata.has_credits(spid)) {
                //printf("ZSQ: !m_sub_partition[%d]->L2_dram_queue_empty() && can_issue_to_dram(%d)\n", spid, spid);
                mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
                m_sub_partition[spid]->L2_dram_queue_pop();
                MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                //printf("ZSQ: sub_partition %d L2_dram_queue to dram_latency_queue, mf sid = %d chip_id = %d sub_partition_id=%u inst @ pc=0x%04x\n", spid,  mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->get_pc());
                fflush(stdout);
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
		//if (mf->get_chip_id()/8 != mf->get_sid()/32) d.ready_cycle += 32; //inter link delay, MCMGPU paper, 32 cycles
                m_dram_latency_queue.push_back(d);
	    	dram_latency_in++;
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.borrow_credit(spid);
                break;  // the DRAM should only accept one request per cycle
            }
    }
#endif

    //kain_NoC_r

    if (!m_dram_latency_queue.empty() &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle)) {
        mem_fetch *mf = m_dram_latency_queue.front().req;

        if (!m_dram->full()) {
            m_dram_latency_queue.pop_front();
            m_dram->push(mf);
        }
    }
}

void memory_partition_unit::set_done( mem_fetch *mf )
{
    unsigned global_spid = mf->get_sub_partition_id(); 
    int spid = global_sub_partition_id_to_local_id(global_spid); 
    assert(m_sub_partition[spid]->get_id() == global_spid); 
    assert( mf->kain_type != CONTEXT_WRITE_REQUEST);
    if (mf->get_access_type() == L1_WRBK_ACC || mf->get_access_type() == L2_WRBK_ACC || mf->kain_type == CONTEXT_WRITE_REQUEST || mf->kain_type == CONTEXT_READ_REQUEST) {
        m_arbitration_metadata.return_credit(spid); 
        MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf, spid); 
    }
    m_sub_partition[spid]->set_done(mf); 
}

//void memory_partition_unit::set_dram_power_stats(unsigned &n_cmd,
//                                                 unsigned &n_activity,
//                                                 unsigned &n_nop,
//                                                 unsigned &n_act,
//                                                 unsigned &n_pre,
//                                                 unsigned &n_rd,
//                                                 unsigned &n_wr,
//                                                 unsigned &n_req) const
//{
//    m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd, n_wr, n_req);
//}


/*void memory_partition_unit::print_stat( FILE * fp ) const
{
    m_dram_r->finish();
    //FIX ME to print the statistics data
}
*/

void memory_partition_unit::print( FILE *fp ) const
{
    fprintf(fp, "Memory Partition %u: \n", m_id); 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->print(fp); 
    }
    
    fprintf(fp, "In Dram Latency Queue (total = %zd): \n", m_dram_latency_queue.size()); 
    for (std::list<dram_delay_t>::const_iterator mf_dlq = m_dram_latency_queue.begin(); 
         mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
        mem_fetch *mf = mf_dlq->req; 
        fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle); 
        if (mf) 
            mf->print(fp); 
        else 
            fprintf(fp, " <NULL mem_fetch?>\n"); 
    }
    
    m_dram->print(fp); 
//	m_dram_r->finish();
}

memory_sub_partition::memory_sub_partition( unsigned sub_partition_id, 
                                            const struct memory_config *config,
                                            class memory_stats_t *stats )
{
    m_id = sub_partition_id;
    m_config=config;
    m_stats=stats;

    assert(m_id < m_config->m_n_mem_sub_partition); 

    char L2c_name[32];
    snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
    m_L2interface = new L2interface(this);
    m_mf_allocator = new partition_mf_allocator(config);

    if(!m_config->m_L2_config.disabled()) {
       m_L2cache = new l2_cache(L2c_name,m_config->m_L2_config,-1,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE);
       m_L2cache->set_sub_partition_id(m_id);
    }

    unsigned int icnt_L2;
    unsigned int L2_dram;
    unsigned int dram_L2;
    unsigned int L2_icnt;
    sscanf(m_config->gpgpu_L2_queue_config,"%u:%u:%u:%u", &icnt_L2,&L2_dram,&dram_L2,&L2_icnt );
    m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2",0,icnt_L2); 
    m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram",0,L2_dram);
    m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2",0,dram_L2);
    m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt",0,L2_icnt);
    wb_addr=-1;
}

memory_sub_partition::~memory_sub_partition()
{
    delete m_icnt_L2_queue;
    delete m_L2_dram_queue;
    delete m_dram_L2_queue;
    delete m_L2_icnt_queue;
    delete m_L2cache;
    delete m_L2interface;
}

//ZSQ data sharing record
std::map<new_addr_type, module_record> record_window_1000; //[i*4+j]: module i's local llc accessed by module j. in mem-side-llc, i is m_id/8  
std::map<new_addr_type, module_record> record_window_5000; //[i*4+j]: module i's local llc accessed by module j. in mem-side-llc, i is m_id/8  
std::map<new_addr_type, module_record> record_window_10000; //[i*4+j]: module i's local llc accessed by module j. in mem-side-llc, i is m_id/8  
std::map<new_addr_type, module_record> record_window_50000; //[i*4+j]: module i's local llc accessed by module j. in mem-side-llc, i is m_id/8  
std::map<new_addr_type, module_record> record_window_100000; //[i*4+j]: module i's local llc accessed by module j. in mem-side-llc, i is m_id/8  
std::map<new_addr_type, module_record> record_total;
std::map<new_addr_type, sharing_record> record_tf;
//ZSQ profile
unsigned long long profile_llc[64][6];
unsigned long long profile_mem[32];
unsigned long long profile_tag[64][16];
unsigned long long profile_hit_miss[64][16][4];

extern unsigned long long llc_w;
extern unsigned long long llc_r;
//unsigned long long kain_request_number = 0;
void memory_sub_partition::cache_cycle( unsigned cycle )
{
    // L2 fill responses
    if( !m_config->m_L2_config.disabled()) {
       if ( m_L2cache->access_ready() && !m_L2_icnt_queue->full() ) {
           mem_fetch *mf = m_L2cache->next_access();
           if(mf->get_access_type() != L2_WR_ALLOC_R){ // Don't pass write allocate read request back to upper level cache
				mf->set_reply();
				mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
               std::ostringstream out;
               unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
               if(gpu_sim_cycle >= 1000000 && gpu_sim_cycle <= 1100000) {
                   out << "L2_icnt_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                       "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                       << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize:"
                       << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tfrom L2\n";
                   std::fstream outdata;
                   outdata.open("report.txt", std::ios_base::app);
                   outdata << out.str().c_str();
                   outdata.close();
               }
				m_L2_icnt_queue->push(mf);
           }else{
				m_request_tracker.erase(mf);
				delete mf;
           }
       }
    }

    // DRAM to L2 (texture) and icnt (not texture)
    if ( !m_dram_L2_queue->empty() ) {
        mem_fetch *mf = m_dram_L2_queue->top();
        if ( mf->kain_type != CONTEXT_READ_REQUEST && !m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf) ) {
            if (m_L2cache->fill_port_free()) {
                mf->set_status(IN_PARTITION_L2_FILL_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_L2cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_dram_L2_queue->pop();
	    	dram_L2_out++;
            }
        }
        else if ( !m_L2_icnt_queue->full() ) {
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            std::ostringstream out;
            unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
            if(gpu_sim_cycle >= 1000000 && gpu_sim_cycle <= 1100000) {
                out << "L2_icnt_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize:"
                    << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tfrom DRAM\n";
                std::fstream outdata;
                outdata.open("report.txt", std::ios_base::app);
                outdata << out.str().c_str();
                outdata.close();
            }
            m_L2_icnt_queue->push(mf);
            m_dram_L2_queue->pop();
	    dram_L2_out++;
        }
    }

    // prior L2 misses inserted into m_L2_dram_queue here
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->cycle();

    // new L2 texture accesses and/or non-texture accesses
    if ( !m_L2_dram_queue->full() && !m_icnt_L2_queue->empty() ) {
        mem_fetch *mf = m_icnt_L2_queue->top();
        if ( (mf->kain_type != CONTEXT_WRITE_REQUEST && mf->kain_type != CONTEXT_READ_REQUEST)&& !m_config->m_L2_config.disabled() &&
              ( (m_config->m_L2_texure_only && mf->istexture()) || (!m_config->m_L2_texure_only) )
           ) {
            // L2 is enabled and access is for L2
            bool output_full = m_L2_icnt_queue->full(); 
            bool port_free = m_L2cache->data_port_free(); 
            if ( !output_full && port_free ) {
                std::list<cache_event> events;
                enum cache_request_status status = m_L2cache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
                bool write_sent = was_write_sent(events);
                bool read_sent = was_read_sent(events);
		
		//0615 sharing counting
		if (status == RESERVATION_FAIL) { // do not record when reservation fail
                    assert(!write_sent);
                    assert(!read_sent);
		} else { //record when hit or miss or reserved

		//ZSQ data sharing record
		bool first_touch;
		unsigned long long last_touch_cycle = -1;
		std::map<unsigned long long,module_record>::iterator it;
	        it = record_window_1000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
       		if (it != record_window_1000.end()) // not first time in this time_window
       		{   
		     first_touch = false;
          	     it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
          	     it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		     it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		     //it->second.access ++;
       		}   
       		else //first time access in this window
       		{   
		    first_touch = true;
		    module_record tmp;
		    for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
			    tmp.record[i][j] = 0;
			}
			for (int j = 0; j < 128; j++) {
			    tmp.record_sm[i][j] = 0;
			}
			it->second.rwtag = 0; //init: read
			it->second.access = 0;
			it->second.hit = 0;
		    }
           	    record_window_1000.insert(std::map<new_addr_type,module_record>::value_type(mf->get_addr()>>7,tmp));
		    it = record_window_1000.find(mf->get_addr()>>7);
		    it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
		    it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		    //it->second.access ++;
       		}
	        it = record_window_5000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
       		if (it != record_window_5000.end()) // not first time in this time_window
       		{   
		     first_touch = false;
          	     it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
          	     it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		     it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		     //it->second.access ++;
       		}   
       		else //first time access in this window
       		{   
		    first_touch = true;
		    module_record tmp;
		    for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
			    tmp.record[i][j] = 0;
			}
			for (int j = 0; j < 128; j++) {
			    tmp.record_sm[i][j] = 0;
			}
			it->second.rwtag = 0; //init: read
			it->second.access = 0;
			it->second.hit = 0;
		    }
           	    record_window_5000.insert(std::map<new_addr_type,module_record>::value_type(mf->get_addr()>>7,tmp));
		    it = record_window_5000.find(mf->get_addr()>>7);
		    it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
		    it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		    //it->second.access ++;
       		}
	        it = record_window_10000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
       		if (it != record_window_10000.end()) // not first time in this time_window
       		{   
		     first_touch = false;
          	     it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
          	     it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		     it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		     //it->second.access ++;
       		}   
       		else //first time access in this window
       		{   
		    first_touch = true;
		    module_record tmp;
		    for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
			    tmp.record[i][j] = 0;
			}
			for (int j = 0; j < 128; j++) {
			    tmp.record_sm[i][j] = 0;
			}
			it->second.rwtag = 0; //init: read
			it->second.access = 0;
			it->second.hit = 0;
		    }
           	    record_window_10000.insert(std::map<new_addr_type,module_record>::value_type(mf->get_addr()>>7,tmp));
		    it = record_window_10000.find(mf->get_addr()>>7);
		    it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
		    it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		    //it->second.access ++;
       		}
	        it = record_window_50000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
       		if (it != record_window_50000.end()) // not first time in this time_window
       		{   
		     first_touch = false;
          	     it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
          	     it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		     it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		     //it->second.access ++;
       		}   
       		else //first time access in this window
       		{   
		    first_touch = true;
		    module_record tmp;
		    for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
			    tmp.record[i][j] = 0;
			}
			for (int j = 0; j < 128; j++) {
			    tmp.record_sm[i][j] = 0;
			}
			it->second.rwtag = 0; //init: read
			it->second.access = 0;
			it->second.hit = 0;
		    }
           	    record_window_50000.insert(std::map<new_addr_type,module_record>::value_type(mf->get_addr()>>7,tmp));
		    it = record_window_50000.find(mf->get_addr()>>7);
		    it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
		    it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		    //it->second.access ++;
       		}
	        it = record_window_100000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
       		if (it != record_window_100000.end()) // not first time in this time_window
       		{   
		     first_touch = false;
          	     it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
          	     it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		     it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		     //it->second.access ++;
       		}   
       		else //first time access in this window
       		{   
		    first_touch = true;
		    module_record tmp;
		    for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
			    tmp.record[i][j] = 0;
			}
			for (int j = 0; j < 128; j++) {
			    tmp.record_sm[i][j] = 0;
			}
			it->second.rwtag = 0; //init: read
			it->second.access = 0;
			it->second.hit = 0;
		    }
           	    record_window_100000.insert(std::map<new_addr_type,module_record>::value_type(mf->get_addr()>>7,tmp));
		    it = record_window_100000.find(mf->get_addr()>>7);
		    it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
		    it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		    //it->second.access ++;
       		}
                it = record_total.find(mf->get_addr()>>7); //block_addr, cache line size 128B
                if (it != record_total.end()) // not first time in this time_window
                {
                     it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
                     it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		     it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		     //it->second.access ++;
                }
                else //first time access in this window
                {
		    module_record tmp;
		    for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
			    tmp.record[i][j] = 0;
			}
			for (int j = 0; j < 128; j++) {
			    tmp.record_sm[i][j] = 0;
			}
			it->second.rwtag = 0; //init: read
			it->second.first_touch = -1; //init: -1 chiplet
			it->second.last_touch_cycle = 0; //init: -1
			it->second.access = 0;
			it->second.hit = 0;
		    }
                    record_total.insert(std::pair<new_addr_type, module_record>(mf->get_addr()>>7,tmp));
		    it = record_total.find(mf->get_addr()>>7);
                    it->second.record[m_id/16][mf->get_sid()/32]++; //record from which module this mf comes
                    it->second.record_sm[m_id/16][mf->get_sid()]++; //record from which module this mf comes
		    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
		    it->second.first_touch = mf->get_sid()/32;
		    //it->second.access ++;
                }	
		last_touch_cycle = it->second.last_touch_cycle;
		it->second.last_touch_cycle = gpu_sim_cycle+gpu_tot_sim_cycle;

                    it = record_window_1000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.access ++;
                    it = record_window_5000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.access ++;
                    it = record_window_10000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.access ++;
                    it = record_window_50000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.access ++;
                    it = record_window_100000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.access ++;
                    it = record_total.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.access ++;
                if ( status == HIT ) {
                    it = record_window_1000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.hit ++;
                    it = record_window_5000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.hit ++;
                    it = record_window_10000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.hit ++;
                    it = record_window_50000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.hit ++;
                    it = record_window_100000.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.hit ++;
                    it = record_total.find(mf->get_addr()>>7); //block_addr, cache line size 128B
		    it->second.hit ++;
		    
                    if( !write_sent ) {
                        // L2 cache replies
                        assert(!read_sent);
                        if( mf->get_access_type() == L1_WRBK_ACC ) {
                            m_request_tracker.erase(mf);
                            delete mf;
                        } else {
                            mf->set_reply();
                            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                            std::ostringstream out;
                            unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
                            if(gpu_sim_cycle >= 1000000 && gpu_sim_cycle <= 1100000) {
                                out << "L2_icnt_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                                    << "\tsize:" << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tcache hit\n";
                                std::fstream outdata;
                                outdata.open("report.txt", std::ios_base::app);
                                outdata << out.str().c_str();
                                outdata.close();
                            }
                            m_L2_icnt_queue->push(mf);
                        }
                        m_icnt_L2_queue->pop();
	    		icnt_L2_out++;
                    } else {
                        assert(write_sent);
                        m_icnt_L2_queue->pop();
	    		icnt_L2_out++;
                    }
		//fprintf(stdout, "LLC access, hit, %llu %llx %llx %s %d %d %s %s %d %llu %llu\n", gpu_sim_cycle+gpu_tot_sim_cycle, mf->get_addr()>>7, mf->get_addr()>>12, mf->is_write()?"R":"W", mf->get_sid()/32, mf->get_chip_id()/8, mf->get_sid()/32 == mf->get_chip_id()/8 ? "L":"R", first_touch?"F":"NF", it->second.first_touch, last_touch_cycle, gpu_sim_cycle+gpu_tot_sim_cycle-last_touch_cycle);
                } else if ( status == MISS) {
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
	    	    icnt_L2_out++;
		//fprintf(stdout, "LLC access, miss, %llu %llx %llx %s %d %d %s %s %d %llu %llu\n", gpu_sim_cycle+gpu_tot_sim_cycle, mf->get_addr()>>7, mf->get_addr()>>12, mf->is_write()?"R":"W", mf->get_sid()/32, mf->get_chip_id()/8, mf->get_sid()/32 == mf->get_chip_id()/8 ? "L":"R", first_touch?"F":"NF", it->second.first_touch, last_touch_cycle, gpu_sim_cycle+gpu_tot_sim_cycle-last_touch_cycle);
                } else if (status == HIT_RESERVED){
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
	    	    icnt_L2_out++;
		//fprintf(stdout, "LLC access, hit_reserved, %llu %llx %llx %s %d %d %s %s %d %llu %llu\n", gpu_sim_cycle+gpu_tot_sim_cycle, mf->get_addr()>>7, mf->get_addr()>>12, mf->is_write()?"R":"W", mf->get_sid()/32, mf->get_chip_id()/8, mf->get_sid()/32 == mf->get_chip_id()/8 ? "L":"R", first_touch?"F":"NF", it->second.first_touch, last_touch_cycle, gpu_sim_cycle+gpu_tot_sim_cycle-last_touch_cycle);
                }
	      
		} //0615 sharing counting //record when hit or miss or reserved
            
	    }
            else
            {
                ;
                //printf("KAIN the port is not free, output full %d, port full %d\n", output_full, port_free); 
                //fflush(stdout);
            }
        }
        else {
            // L2 is disabled or non-texture access to texture-only L2
            mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_dram_queue->push(mf);
	    L2_dram_in++;
        std::ostringstream out;
            unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
            if(gpu_sim_cycle >= 1000000  && gpu_sim_cycle <= 1100000) {
                out << "L2_DRAM_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                    << "\tsize:" << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tcache miss\n";
                std::fstream outdata;
                outdata.open("report.txt", std::ios_base::app);
                outdata << out.str().c_str();
                outdata.close();
            }
            m_icnt_L2_queue->pop();
	    icnt_L2_out++;
        }
    }
	else
	{
        ;
	//	if(m_L2_dram_queue->full() && (gpu_sim_cycle+gpu_tot_sim_cycle)%1 == 0)	
	//		printf("KAIN m_L2_dram_queue is full, id %d\n", m_id);
	//	if(m_icnt_L2_queue->empty() && (gpu_sim_cycle+gpu_tot_sim_cycle)%1 == 0)	
	//		printf("KAIN m_icnt_L2 queue is empty \n");
	}

    // ROP delay queue
    if( !m_rop.empty() && (cycle >= m_rop.front().ready_cycle) && !m_icnt_L2_queue->full() ) {
        mem_fetch* mf = m_rop.front().req;
	if (mf->is_write()) llc_w++;
	else llc_r++;
//		if(mf->kain_type == CONTEXT_WRITE_REQUEST)
//			printf("KAIN received the write reuquest %lld\n",kain_request_number++);
        m_rop.pop();
	rop_out++;
        m_icnt_L2_queue->push(mf);
	icnt_L2_in++;
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
    }
}

bool memory_sub_partition::full() const
{
    return m_icnt_L2_queue->full();
}

bool memory_sub_partition::L2_dram_queue_empty() const
{
   return m_L2_dram_queue->empty(); 
}

class mem_fetch* memory_sub_partition::L2_dram_queue_top() const
{
   return m_L2_dram_queue->top(); 
}

void memory_sub_partition::L2_dram_queue_pop() 
{
   m_L2_dram_queue->pop(); 
   L2_dram_out++;
}

bool memory_sub_partition::dram_L2_queue_full() const
{
   return m_dram_L2_queue->full(); 
}

void memory_sub_partition::dram_L2_queue_push( class mem_fetch* mf )
{
   m_dram_L2_queue->push(mf); 
   dram_L2_in++;
}

void memory_sub_partition::print_cache_stat(unsigned &accesses, unsigned &misses) const
{
    FILE *fp = stdout;
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->print(fp,accesses,misses);
}

void memory_sub_partition::print( FILE *fp ) const
{

/*    if ( !m_request_tracker.empty() ) {
        fprintf(fp,"Memory Sub Parition %u: pending memory requests:\n", m_id);
        for ( std::set<mem_fetch*>::const_iterator r=m_request_tracker.begin(); r != m_request_tracker.end(); ++r ) {
            mem_fetch *mf = *r;
            if ( mf )
                mf->print(fp);
            else
                fprintf(fp," <NULL mem_fetch?>\n");
        }
    }
*/
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print( gzFile visualizer_file )
{
   // gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
   // gzprintf(visualizer_file, "Ltwowritehit: %d\n",  L2_write_access-L2_write_miss);
   // gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
   // gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_access-L2_read_miss);
   if (num_mfs)
      gzprintf(visualizer_file, "averagemflatency: %lld\n", mf_total_lat/num_mfs);
}

void gpgpu_sim::print_dram_stats(FILE *fout) const
{
	unsigned cmd=0;
	unsigned activity=0;
	unsigned nop=0;
	unsigned act=0;
	unsigned pre=0;
	unsigned rd=0;
	unsigned wr=0;
	unsigned req=0;
	unsigned tot_cmd=0;
	unsigned tot_nop=0;
	unsigned tot_act=0;
	unsigned tot_pre=0;
	unsigned tot_rd=0;
	unsigned tot_wr=0;
	unsigned tot_req=0;

	for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
//		m_memory_partition_unit[i]->set_dram_power_stats(cmd,activity,nop,act,pre,rd,wr,req);
		tot_cmd+=cmd;
		tot_nop+=nop;
		tot_act+=act;
		tot_pre+=pre;
		tot_rd+=rd;
		tot_wr+=wr;
		tot_req+=req;
	}
    fprintf(fout,"gpgpu_n_dram_reads = %d\n",tot_rd );
    fprintf(fout,"gpgpu_n_dram_writes = %d\n",tot_wr );
    fprintf(fout,"gpgpu_n_dram_activate = %d\n",tot_act );
    fprintf(fout,"gpgpu_n_dram_commands = %d\n",tot_cmd);
    fprintf(fout,"gpgpu_n_dram_noops = %d\n",tot_nop );
    fprintf(fout,"gpgpu_n_dram_precharges = %d\n",tot_pre );
    fprintf(fout,"gpgpu_n_dram_requests = %d\n",tot_req );
}

unsigned memory_sub_partition::flushL2() 
{ 
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->flush(); 
    }
    return 0; // L2 is read only in this version
}

bool memory_sub_partition::busy() const 
{
    return !m_request_tracker.empty();
}

void memory_sub_partition::push( mem_fetch* req, unsigned long long cycle ) 
{
    if (req) {
        m_request_tracker.insert(req);
        m_stats->memlatstat_icnt2mem_pop(req);
        if( req->istexture() ) {
            m_icnt_L2_queue->push(req);
	    icnt_L2_in++;
            req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        } else {
            rop_delay_t r;
            r.req = req;
            r.ready_cycle = cycle + m_config->rop_latency;
            m_rop.push(r);
	    rop_in++;
            req->set_status(IN_PARTITION_ROP_DELAY,gpu_sim_cycle+gpu_tot_sim_cycle);
	    //printf("ZSQ: cycle %llu, m_rop.push, ", gpu_sim_cycle+gpu_tot_sim_cycle);
	    //req->mf_print();
        }
    }
}

mem_fetch* memory_sub_partition::pop() 
{
    mem_fetch* mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if ( mf && mf->isatomic() )
        mf->do_atomic();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        delete mf;
        mf = NULL;
    } 
    return mf;
}

mem_fetch* memory_sub_partition::top() 
{
    mem_fetch *mf = m_L2_icnt_queue->top();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        m_L2_icnt_queue->pop();
        m_request_tracker.erase(mf);
        delete mf;
        mf = NULL;
    } 
    return mf;
}

void memory_sub_partition::set_done( mem_fetch *mf )
{
    m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(class cache_stats &l2_stats) const {
    if (!m_config->m_L2_config.disabled()) {
        l2_stats += m_L2cache->get_stats();
    }
}

void memory_sub_partition::get_L2cache_sub_stats(struct cache_sub_stats &css) const{
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->get_sub_stats(css);
    }
}

void memory_sub_partition::get_L2cache_sub_stats_kain(unsigned cluster_id, struct cache_sub_stats &css) const{
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->get_sub_stats_kain(cluster_id,css);
    }
}


void memory_sub_partition::clear_L2cache_sub_stats_kain() {
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->clear_sub_stats_kain();
    }
}

void memory_sub_partition::visualizer_print( gzFile visualizer_file )
{
    // TODO: Add visualizer stats for L2 cache
}

void
memory_sub_partition::set_mk_scheduler(MKScheduler* mk_sched)
{
    if (!m_config->m_L2_config.disabled()) {
    m_L2cache->set_mk_scheduler(mk_sched);
    }
}

