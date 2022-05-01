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
#include <sstream>
#include <list>
#include <set>

#include "../option_parser.h"
#include "mem_fetch.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"
#include "shader.h"
#include "mem_latency_stat.h"
#include "l2cache_trace.h"
#include "../ramulator_sim/Config.h"

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



mem_fetch *partition_mf_allocator::alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr) const {
    assert(wr);
    mem_access_t access(type, addr, size, wr);
    mem_fetch *mf = new mem_fetch(access,
                                  NULL,
                                  WRITE_PACKET_SIZE,
                                  -1,
                                  -1,
                                  -1,
                                  m_memory_config);
    return mf;
}

new_addr_type *kain_cache[4];
static int kain_init = 0;

memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const struct memory_config *config,
                                             class memory_stats_t *stats)
        : m_id(partition_id), m_config(config), m_stats(stats), m_arbitration_metadata(config) {
//    m_dram = new dram_t(m_id,m_config,m_stats,this);
    Config m_r_config("HBM-config.cfg");
    m_r_config.set_core_num(core_numbers);

    m_dram_r = new GpuWrapper(m_r_config, m_config->m_L2_config.get_line_sz(), this, m_id);

    if (kain_init == 0) {
        kain_init = 1;
        for (int j = 0; j < 4; j++) {
            kain_cache[j] = new new_addr_type[8388608];//1024*1024
            for (unsigned i = 0; i < 8388608; i++)
                kain_cache[j][i] = 0;
        }
    }

    m_sub_partition = new memory_sub_partition *[m_config->m_n_sub_partition_per_memory_channel];
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned sub_partition_id = m_id * m_config->m_n_sub_partition_per_memory_channel + p;
        m_sub_partition[p] = new memory_sub_partition(sub_partition_id, m_config, stats);
    }
}

memory_partition_unit::~memory_partition_unit() {
//    delete m_dram; 
    delete m_dram_r;
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        delete m_sub_partition[p];
    }
    delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(const struct memory_config *config)
        : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
          m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
          m_shared_credit(0) {
    // each sub partition get at least 1 credit for forward progress 
    // the rest is shared among with other partitions 
    m_private_credit_limit = 1;
    m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size
                            + config->gpgpu_dram_return_queue_size
                            - (config->m_n_sub_partition_per_memory_channel - 1);
    if (config->gpgpu_frfcfs_dram_sched_queue_size == 0
        or config->gpgpu_dram_return_queue_size == 0) {
        m_shared_credit_limit = 0; // no limit if either of the queue has no limit in size 
    }
    assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(int inner_sub_partition_id) const {
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
void memory_partition_unit::arbitration_metadata::borrow_credit(int inner_sub_partition_id) {
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

void memory_partition_unit::arbitration_metadata::return_credit(int inner_sub_partition_id) {
    int spid = inner_sub_partition_id;
    if (m_private_credit[spid] > 0) {
        m_private_credit[spid] -= 1;
    } else {
        m_shared_credit -= 1;
    }
    assert((m_shared_credit >= 0) && "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
    fprintf(fp, "private_credit = ");
    for (unsigned p = 0; p < m_private_credit.size(); p++) {
        fprintf(fp, "%d ", m_private_credit[p]);
    }
    fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
    fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit, m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
    bool busy = false;
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        if (m_sub_partition[p]->busy()) {
            busy = true;
        }
    }
    return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->cache_cycle(cycle);
    }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
/*
    m_dram->visualizer_print(visualizer_file);
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->visualizer_print(visualizer_file); 
    }
*/
}

// determine whether a given subpartition can issue to DRAM 
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
    int spid = inner_sub_partition_id;
    bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
    bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

    MEMPART_DPRINTF("sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
                    spid, (sub_partition_contention) ? 'T' : 'F', (has_dram_resource) ? 'T' : 'F');

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
             d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
             m_dram_latency_queue.push_back(d);
            if(gpu_sim_cycle > 1000000) {
                out << "dram push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type() << "\tcycle: " <<
                    ::_get_icnt_cycle() << "\tchip: " << mf->get_sid() / 32 << "\tsize: " << packet_size
                    <<"\tgpu_cycle: " << gpu_sim_cycle << "\n";
                rep1->apply(out.str().c_str());
            }
         dram_latency_in++;
             mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
}
#endif

int memory_partition_unit::global_sub_partition_id_to_local_id(int global_sub_partition_id) const {
    return (global_sub_partition_id - m_id * m_config->m_n_sub_partition_per_memory_channel);
}


class KAIN_GPU_chiplet KAIN_NoC_r(3);

long long KAIN_HBM_Cache_hit = 0;
long long KAIN_HBM_Cache_miss = 0;

int kain_memory_page_count[4];
extern std::map<new_addr_type, new_addr_type *> KAIN_page_table;
#if HBM_CACHE == 1
std::list<mem_fetch*> KAIN_HBM_Cache_request[32];
#endif

void memory_partition_unit::dram_cycle() {
/*
    // pop completed memory request from dram and push it to dram-to-L2 queue 
    // of the original sub partition 
    mem_fetch* mf_return = m_dram->return_queue_top();
    if (mf_return) {
        unsigned dest_global_spid = mf_return->get_sub_partition_id(); 
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid); 
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid); 
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            if( mf_return->get_access_type() == L1_WRBK_ACC ) {
                m_sub_partition[dest_spid]->set_done(mf_return); 
                delete mf_return;
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.return_credit(dest_spid); 
                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid); 
            }
            m_dram->return_queue_pop(); 
        }
    } else {
        m_dram->return_queue_pop(); 
    }
    
    m_dram->cycle(); 
    m_dram->dram_log(SAMPLELOG);   

    if( !m_dram->full() ) {
        // L2->DRAM queue to DRAM latency queue
        // Arbitrate among multiple L2 subpartitions 
        int last_issued_partition = m_arbitration_metadata.last_borrower(); 
        for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
            int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel; 
            if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid)) {
                mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
                m_sub_partition[spid]->L2_dram_queue_pop();
                MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid); 
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                m_dram_latency_queue.push_back(d);
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.borrow_credit(spid); 
                break;  // the DRAM should only accept one request per cycle 
            }
//			else
//			{
//				if(!can_issue_to_dram(spid))	
//					if( (gpu_sim_cycle+gpu_tot_sim_cycle)%10000 == 0)	
//						printf("KAIN cannot issue to dram\n");
//			}

        }
    }
//	else
//	{
//		if( (gpu_sim_cycle+gpu_tot_sim_cycle)%10000 == 0)	
//			printf("KAIN m_dram is full\n");
//	}

    // DRAM latency queue
    if( !m_dram_latency_queue.empty() && ( (gpu_sim_cycle+gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle ) && !m_dram->full() ) {
        mem_fetch* mf = m_dram_latency_queue.front().req;
        m_dram_latency_queue.pop_front();
        m_dram->push(mf);
    }
*/

/* ZSQ 20210130 Rearranged in the latter piece of code
#if SM_SIDE_LLC == 1
unsigned _mid = m_id;
unsigned _subid = _mid*2;
if(!KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid)) {

    mem_fetch* mf_return = m_dram_r->r_return_queue_top();
    if (mf_return) {
//ZSQ0123
//	if( mf_return->get_access_type() == L1_WRBK_ACC ) {
//		delete mf_return;
//		m_dram_r->r_return_queue_pop();
//	}
//	else 
//ZSQ0123
	     if (mf_return->get_sid()/32 != mf_return->get_chip_id()/8) { //remote, push to inter_icnt
		unsigned to_module = 192 + mf_return->get_sid()/32;
		unsigned from_module = 192 + mf_return->get_chip_id()/8;
		//ZSQ0126
		if (INTER_TOPO == 1 && (mf_return->get_sid()/32+mf_return->get_chip_id()/8)%2 == 0) //ring, forward
		    to_module = 192 + (mf_return->get_sid()/32+1)%4;
		//ZSQ0126
		unsigned response_size = mf_return->get_is_write()?mf_return->get_ctrl_size():mf_return->size();
		//printf("ZSQ: inter-monule icnt_push() from %d to %d, mf is %s NULL, sid = %d chip_id = %d sub_partition_id=%u type = %d inst @ pc=0x%04x\n", from_module, to_module, mf_return?"not":"", mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id(), (int*)mf_return->get_type(), mf_return->get_pc()); 
		//fflush(stdout);
		if (::icnt_has_buffer(from_module, response_size)) {
		    ::icnt_push(from_module, to_module, (void*)mf_return, response_size);
		    m_dram_r->r_return_queue_pop();
		}
	}
	else { //local, push to dram_L2_queue
	        unsigned dest_global_spid = mf_return->get_sub_partition_id();
        	int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        	assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
        	if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            		//ZSQ0123 if( mf_return->get_access_type() == L1_WRBK_ACC ) {
		        if(mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {//ZSQ0123
                	m_sub_partition[dest_spid]->set_done(mf_return);
                	delete mf_return;
            		} else {
                		m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                		mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                		m_arbitration_metadata.return_credit(dest_spid);
                		MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
            		}
            	m_dram_r->r_return_queue_pop();
        	}
	}
KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid);
    } else {
        mf_return = m_dram_r->r_return_queue_pop();
	int pop_flag = -1;
  if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid+1)) {
        if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) 
  	    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
	if (mf_return) {
	    KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
	    pop_flag = 1;
	}
	else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
	    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
	    if (mf_return) pop_flag = 0;
	}	
  } else {
	if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
	    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
	if (mf_return) {
	    KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
	    pop_flag = 0;
	}
	else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) {
	    mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
	    if (mf_return) pop_flag = 1;
	}
  }
  if (mf_return) {
  unsigned dest_global_spid = mf_return->get_sub_partition_id();
  int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
  if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                    m_arbitration_metadata.return_credit(dest_spid); //ZSQ0123
                    //ZSQ0123 if( mf_return->get_access_type() == L1_WRBK_ACC ) {
		    if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) { //ZSQ0123
			m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                    }
    if (pop_flag == 0)
        KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
    else if (pop_flag == 1)
	KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
  }
  } else {
//ZSQ0123
if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
//ZSQ0123
}
    }
} else {
mem_fetch* mf_return = NULL;
  int pop_flag = -1;
  if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid+1)) {
        if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1))
            mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
        if (mf_return) {
            KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
            pop_flag = 1;
        }
        else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
            mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
            if (mf_return) pop_flag = 0;
        }
  } else {
        if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
            mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid);
        if (mf_return) {
            KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
            pop_flag = 0;
        }
        else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) {
            mf_return = KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1);
            if (mf_return) pop_flag = 1;
        }
  }
  if (mf_return) {
                    unsigned dest_global_spid = mf_return->get_sub_partition_id();
                    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
  		if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
		    m_arbitration_metadata.return_credit(dest_spid); //ZSQ0123
                    //ZSQ0123 if( mf_return->get_access_type() == L1_WRBK_ACC ) {
		    if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) { //ZSQ0123
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                    }
		}
    if (pop_flag == 0)
        KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
    else if (pop_flag == 1)
        KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);

  KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid);
  } else {
    //ZSQ0123
    if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
    if (KAIN_NoC_r.inter_icnt_pop_llc_top(_subid+1)==NULL) KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
    //ZSQ0123
	
    mf_return = m_dram_r->r_return_queue_top();
    if (mf_return) {
//ZSQ0123
//        if( mf_return->get_access_type() == L1_WRBK_ACC ) {
//                delete mf_return;
//                m_dram_r->r_return_queue_pop();
//        }
//        else 
//ZSQ0123
	     if (mf_return->get_sid()/32 != mf_return->get_chip_id()/8) { //remote, push to inter_icnt
                unsigned to_module = 192 + mf_return->get_sid()/32;
                unsigned from_module = 192 + mf_return->get_chip_id()/8;
		//ZSQ0126
		if (INTER_TOPO == 1 && (mf_return->get_sid()/32+mf_return->get_chip_id()/8)%2 == 0) //ring, forward
		    to_module = 192 + (mf_return->get_sid()/32+1)%4;
		//ZSQ0126
                unsigned response_size = mf_return->get_is_write()?mf_return->get_ctrl_size():mf_return->size();
                //printf("ZSQ: inter-monule icnt_push() from %d to %d, mf is %s NULL, sid = %d chip_id = %d sub_partition_id=%u type = %d inst @ pc=0x%04x\n", from_module, to_module, mf_return?"not":"", mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id(), (int*)mf_return->get_type(), mf_return->get_pc());
                //fflush(stdout);
                if (::icnt_has_buffer(from_module, response_size)) {
                    ::icnt_push(from_module, to_module, (void*)mf_return, response_size);
                    m_dram_r->r_return_queue_pop();
                }
        }
        else { //local, push to dram_L2_queue
                unsigned dest_global_spid = mf_return->get_sub_partition_id();
                int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
                if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                        //ZSQ0123 if( mf_return->get_access_type() == L1_WRBK_ACC ) {
		 	if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) { //ZSQ0123
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                        } else {
                                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                                m_arbitration_metadata.return_credit(dest_spid);
                                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
                        }
                m_dram_r->r_return_queue_pop();
                }
        }
    
    } else m_dram_r->r_return_queue_pop();
  }

}
#endif
ZSQ 20210130 Rearranged in the latter piece of code */

//ZSQ 20210130 Rearranged the above piece of code here
#if SM_SIDE_LLC == 1
    unsigned _mid = m_id;
    unsigned _subid = _mid*2;
    if(!KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid)) {   //returnq turn, start
        mem_fetch* mf_return = m_dram_r->r_return_queue_top();
        inter_delay_t *x4;
    //printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle(), ->dram_L2_queue, returnq turn, r_return_queue_top()\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id);
        if (mf_return) {    //returnq turn, m_dram_r->r_return_queue_top() != NULL, start
    //printf("	!NULL, mf sid = %d, chip_id = %d, sub_id = %d\n", mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id());
            if (mf_return->get_sid()/32 != mf_return->get_chip_id()/8) { //remote, push to inter_icnt
                unsigned to_module = 192 + mf_return->get_sid()/32;
                unsigned from_module = 192 + mf_return->get_chip_id()/8;
                mf_return->set_src(from_module);
                mf_return->set_dst(to_module);
                if (INTER_TOPO == 1 && (mf_return->get_sid()/32+mf_return->get_chip_id()/8)%2 == 0) {//ring, forward
                    to_module = 192 + (mf_return->get_sid()/32+1)%4;
                    mf_return->set_next_hop(to_module);
                }
                unsigned response_size = mf_return->get_is_write()?mf_return->get_ctrl_size():mf_return->size();
                if (::icnt_has_buffer(from_module, response_size)) {
                    ::icnt_push(from_module, to_module, (void*)mf_return, response_size);
                    m_dram_r->r_return_queue_pop();
                    returnq_out++;
                    returnq_out_inter++;
                }
            }
            else { //local, push to dram_L2_queue
                unsigned dest_global_spid = mf_return->get_sub_partition_id();
                int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
                if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                    if(mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                returnq_out_delete++;
                printf("ZSQ: should not arrive here.\n");
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                        m_arbitration_metadata.return_credit(dest_spid);
                        MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
                    }
                    m_dram_r->r_return_queue_pop();
            returnq_out++;
            returnq_out_local++;
    //printf("		local, WRBK delete or ->dram_L2_queue, dest_global_spid = %d, dest_spid = %d\n", dest_global_spid, dest_spid);
                }
            }
            KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid);
        }   //returnq turn, m_dram_r->r_return_queue_top() != NULL, end
        else {    //returnq turn, m_dram_r->r_return_queue_top() = NULL, start
            m_dram_r->r_return_queue_pop();
    //printf("        NULL, returnq size = %d\n", m_dram_r->r_returnq_size());
        if (!m_sub_partition[0]->dram_L2_queue_full()&&!m_sub_partition[1]->dram_L2_queue_full()) {
            if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid+1)) {
            if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) {
                x4 = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
                mf_return = x4->req;
                mf_return->set_icnt_cycle(x4->ready_cycle);
                KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
            } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
                x4 = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
                mf_return = x4->req;
                mf_return->set_icnt_cycle(x4->ready_cycle);
            } else {
            if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
                        x4 = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid);
                        mf_return = x4->req;
                        mf_return->set_icnt_cycle(x4->ready_cycle);
                        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
                    } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1))
                        x4 = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1);
                        mf_return = x4->req;
                        mf_return->set_icnt_cycle(x4->ready_cycle);
            }
            if (mf_return) {
            unsigned dest_global_spid = mf_return->get_sub_partition_id();
            int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
            m_arbitration_metadata.return_credit(dest_spid);
                    if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                        returnq_out_delete++;
                        returnq_out_inter_pop_delete++;
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
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
        mem_fetch* mf_return = NULL;
        bool flag = false;
            if (!m_sub_partition[0]->dram_L2_queue_full()&&!m_sub_partition[1]->dram_L2_queue_full()) {
                if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(_subid+1)) {
                    if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1)) {
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1)->req;
                        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
                    } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid))
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid)->req;
                } else {
                    if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid)) {
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid)->req;
                        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(_subid+1);
                    } else if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(_subid+1))
                        mf_return = KAIN_NoC_r.inter_icnt_pop_llc_pop(_subid+1)->req;
                }
                if (mf_return) {
                    flag = true;
                    unsigned dest_global_spid = mf_return->get_sub_partition_id();
                    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                    m_arbitration_metadata.return_credit(dest_spid);
                    if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                        m_sub_partition[dest_spid]->set_done(mf_return);
                        delete mf_return;
                        returnq_out_delete++;
                        returnq_out_inter_pop_delete++;
                    } else {
                        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
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
        if(!flag) {
            mf_return = m_dram_r->r_return_queue_top();
    //printf("ZSQ: cycle %llu, mem_partition %d, dram_cycle(), ->dram_L2_queue, inter_llc turn, r_return_queue_top()\n", gpu_sim_cycle+gpu_tot_sim_cycle, m_id);
            if (mf_return) {
    //printf("	!NULL, mf sid = %d, chip_id = %d, sub_id = %d\n", mf_return->get_sid(), mf_return->get_chip_id(), mf_return->get_sub_partition_id());
                if (mf_return->get_sid()/32 != mf_return->get_chip_id()/8) { //remote, push to inter_icnt
                    unsigned to_module = 192 + mf_return->get_sid()/32;
                    unsigned from_module = 192 + mf_return->get_chip_id()/8;
                    mf_return->set_src(from_module);
                    mf_return->set_dst(to_module);
                    mf_return->set_next_hop(to_module);

                    if (INTER_TOPO == 1 && (mf_return->get_sid()/32+mf_return->get_chip_id()/8)%2 == 0) //ring, forward
                        to_module = 192 + (mf_return->get_sid()/32+1)%4;
                    unsigned response_size = mf_return->get_is_write()?mf_return->get_ctrl_size():mf_return->size();
                    if (::icnt_has_buffer(from_module, response_size)) {
                        ::icnt_push(from_module, to_module, (void*)mf_return, response_size);
                        m_dram_r->r_return_queue_pop();
                returnq_out++;
                returnq_out_inter++;
    //printf("		remote, icnt_push(%d, %d)\n", from_module, to_module);
                    }
                } else { //local, push to dram_L2_queue
                    unsigned dest_global_spid = mf_return->get_sub_partition_id();
                    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
                    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
                    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
                        if( mf_return->get_access_type() == L1_WRBK_ACC || mf_return->get_access_type() == L2_WRBK_ACC) {
                            m_sub_partition[dest_spid]->set_done(mf_return);
                            delete mf_return;
                    returnq_out_delete++;
                printf("ZSQ: should not arrive here.\n");
    //printf("                local, error\n");
                        } else {
                            m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                            mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                            m_arbitration_metadata.return_credit(dest_spid);
                            MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
    //printf("		local, ->dram_L2_queue, dest_global_spid = %d, dest_spid = %d\n", dest_global_spid, dest_spid);
                        }
                        m_dram_r->r_return_queue_pop();
                returnq_out++;
                returnq_out_local++;
                    }
                }
            } else {
                 m_dram_r->r_return_queue_pop();
            }
        } //inter_icnt_pop_llc turn, top() = NULL, end
    }   //inter_icnt_pop_llc turn, end
#endif
//ZSQ 20210130 Rearranged the above piece of code here


#if SM_SIDE_LLC == 0
    mem_fetch *mf_return = m_dram_r->r_return_queue_top();
#if BEN_OUTPUT == 1
    std::ostringstream out;
#endif
    if (mf_return) {
        unsigned dest_global_spid = mf_return->get_sub_partition_id();
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
        if (!m_sub_partition[0]->dram_L2_queue_full() && !m_sub_partition[1]->dram_L2_queue_full()) {
            if (mf_return->get_access_type() == L1_WRBK_ACC) {
                m_sub_partition[dest_spid]->set_done(mf_return);
                delete mf_return;
                returnq_out_delete++;
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                m_arbitration_metadata.return_credit(dest_spid);
                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
            }
            m_dram_r->r_return_queue_pop();
            returnq_out++;
        }
    } else {
        m_dram_r->r_return_queue_pop();
    }
#endif
/*
    mem_fetch* mf_return = m_dram_r->r_return_queue_top();
    if (mf_return) {
            //printf("Reply, mf id %u, tpc_id %d, mf->chip_id %d, current mid %d, dest_id %d\n", mf_return->get_request_uid(),mf_return->get_tpc(),mf_return->get_chip_id(), m_id, (mf_return->get_tpc()/64)*2+(((mf_return->bankID())& 0x1f)/16));
            //fflush(stdout);
            //((tlx->bk)& 0xf) << 1 + tlx->col & 0x1
            //
            //
            bool HBM_cache = mf_return->get_access_type() == GLOBAL_ACC_R || mf_return->get_access_type() == GLOBAL_ACC_W;
            //HBM_cache = false;
            if(HBM_cache == false || mf_return->get_chip_id() == m_id || mf_return->kain_miss_HBM_cache == 1)//belong to this, not from HBM Caching
            {
                assert(mf_return->kain_type != CONTEXT_WRITE_REQUEST);
                if(!KAIN_NoC_r.reply_full(mf_return, (mf_return->get_sub_partition_id()/2)/8, m_id/8))//SM0-20 use MC0 16 LLC slices
                {
                    KAIN_NoC_r.reply_push(mf_return, (mf_return->get_sub_partition_id()/2)/8, m_id/8);//SM0-20 use MC0 16 LLC slices, SM20-40 use MC1
                    m_dram_r->r_return_queue_pop();

                }
            }
            else// this is from HBM Cache, need to check write ack or read (in HBM Cache hit or not)
            {
                if(mf_return->kain_type == CONTEXT_WRITE_REQUEST)//This is the write reply from writing the HBM Cache
                {
                    new_addr_type addr = (mf_return->kain_get_addr())>>7;
                    kain_cache[(m_id/8)][(mf_return->get_addr()>>7)%8388608] = (mf_return->get_addr()>>7);//fill the HBM CAche
                  //  printf("write the data addr %d, Location %d, chip id %d-%d, addr %0x\n", addr, m_id/8, mf_return->get_chip_id(), mf_return->get_chip_id()%8, (mf_return->kain_get_addr()>>7));
                  //  fflush(stdout);

                    m_dram_r->r_return_queue_pop();
                    delete mf_return;
                }
                //TO DO
                //if Miss, send to the HBM remote to fetch the data
                else
                {

                    //HBM_Cache_hit = true;

                    if(mf_return->kain_HBM_Cache_hit_miss == 1)//hit
                    {
                        if(!KAIN_NoC_r.reply_full(mf_return, (mf_return->get_sub_partition_id()/2)/8, m_id/8))//SM0-20 use MC0 16 LLC slices
                        {
                            KAIN_NoC_r.reply_push(mf_return, (mf_return->get_sub_partition_id()/2)/8, m_id/8);//SM0-20 use MC0 16 LLC slices, SM20-40 use MC1
                            m_dram_r->r_return_queue_pop();
                            KAIN_HBM_Cache_hit++;
                        }
                    }
                    if(mf_return->kain_HBM_Cache_hit_miss == 0)//miss
                    {
                        //printf("KAIN, HBM miss come here\n");
                        m_dram_r->r_return_queue_pop();
                        delete mf_return;
                        KAIN_HBM_Cache_miss++;
                    }
                }
            }
    } else {
        m_dram_r->r_return_queue_pop();
    }
*/
/*
    mf_return = KAIN_NoC_r.reply_top(m_id);

#if HBM_CACHE == 1
    bool HBM_cache_request_full = false;
    if(mf_return && mf_return->kain_HBM_cache_channel != -1)
        if(KAIN_HBM_Cache_request[(m_id/8)*8+mf_return->kain_HBM_cache_channel].size()>=256)
            HBM_cache_request_full = true;

    if (mf_return && HBM_cache_request_full == false) {

             	    if(mf_return->kain_miss_HBM_cache == 1)//This is the HBM miss reply from remote memory, need to write into HBM Cache
                    {
                        //KAIN TODO, this addrr needs to be changed, the same with the read HBM Cache addr
                        mem_access_t access( GLOBAL_ACC_W,mf_return->get_addr(), 128, 1); 
                        mem_fetch *mf = new mem_fetch( access,
                        NULL,
                        32, // flit size 32
                        -1,  
                        mf_return->get_sid(), 
                        mf_return->get_tpc(),
                        mf_return->get_mem_config());

                        mf->kain_type = CONTEXT_WRITE_REQUEST;
                        mf->kain_stream_id = mf_return->kain_stream_id;

                        mf->kain_transform_to_HBM_Cache_address();
                        KAIN_HBM_Cache_request[(m_id/8)*8+mf->kain_HBM_cache_channel].push_back(mf); 
                    }
#endif

#if HBM_CACHE == 0
    if (mf_return) {
#endif
        unsigned dest_global_spid = mf_return->get_sub_partition_id();
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            if ( mf_return->get_access_type() == L1_WRBK_ACC ) { 
                m_sub_partition[dest_spid]->set_done(mf_return);
                delete mf_return;
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                m_arbitration_metadata.return_credit(dest_spid);
                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid);
            }
            KAIN_NoC_r.reply_pop_front(m_id);
        }
    } else if (mf_return == NULL){
        KAIN_NoC_r.reply_pop_front(m_id);
    }
*/
//    printf("KAIN into cycle\n");
//    fflush(stdout);
    m_dram_r->cycle(); // In this part, when read/write complete, the return q should be automatically written due to the call back function.
//    printf("KAIN out cycle\n");
//    fflush(stdout);

/* ZSQ 20210130 Rearranged in the latter piece of code
#if SM_SIDE_LLC == 1
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;

if(!KAIN_NoC_r.get_inter_icnt_pop_mem_turn(m_id)){

        if (!m_sub_partition[spid]->L2_dram_queue_empty()) {
            mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();

	    if (mf->get_sid()/32 != mf->get_chip_id()/8){ //remote, push to inter_icnt
		unsigned from_module = 192 + mf->get_sid()/32;
                unsigned to_module = 192 + mf->get_chip_id()/8;

		//ZSQ0126
		if (INTER_TOPO == 1 && (mf->get_sid()/32+mf->get_chip_id()/8)%2 == 0) //ring, forward
		    to_module = 192 + (mf->get_chip_id()/8+1)%4;
		//ZSQ0126

                unsigned size = mf->get_is_write()?mf->size():mf->get_ctrl_size();
		//printf("ZSQ: inter-module icnt_push() from %d to %d, mf is %s NULL, sid = %d chip_id = %d sub_partition_id=%u type = %d inst @ pc=0x%04x\n", from_module, to_module, mf?"not":"", mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), (int*)mf->get_type(), mf->get_pc()); 
		//fflush(stdout);
		if (::icnt_has_buffer(from_module, size)) {
                    ::icnt_push(from_module, to_module, (void*)mf, size);
		    m_sub_partition[spid]->L2_dram_queue_pop();
		    //ZSQ0123
		    m_arbitration_metadata.borrow_credit(spid);
		}
       	    }
	    else if (can_issue_to_dram(spid)){ //local. push to dram_latency_queue
	
                m_sub_partition[spid]->L2_dram_queue_pop();
                MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                //printf("ZSQ: sub_partition %d L2_dram_queue to dram_latency_queue, mf sid = %d chip_id = %d sub_partition_id=%u inst @ pc=0x%04x\n", spid,  mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->get_pc());
                fflush(stdout);
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                m_dram_latency_queue.push_back(d);
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.borrow_credit(spid);
                break;  // the DRAM should only accept one request per cycle
            }
KAIN_NoC_r.set_inter_icnt_pop_mem_turn(m_id);
    	}
else {
  if (!KAIN_NoC_r.inter_icnt_pop_mem_empty(m_id)) {
  mem_fetch *mf =  KAIN_NoC_r.inter_icnt_pop_mem_pop(m_id);
  dram_delay_t d;
  d.req = mf;
  d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency+32;
  m_dram_latency_queue.push_back(d);
  mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
  }
}
}else {
  if (!KAIN_NoC_r.inter_icnt_pop_mem_empty(m_id)) {
  mem_fetch *mf =  KAIN_NoC_r.inter_icnt_pop_mem_pop(m_id);
  dram_delay_t d;
  d.req = mf;
  d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency+32;
  m_dram_latency_queue.push_back(d);
  mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
  KAIN_NoC_r.set_inter_icnt_pop_mem_turn(m_id);
  } else {
	    if (!m_sub_partition[spid]->L2_dram_queue_empty()) {
            mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();

            if (mf->get_sid()/32 != mf->get_chip_id()/8){ //remote, push to inter_icnt
                unsigned from_module = 192 + mf->get_sid()/32;
                unsigned to_module = 192 + mf->get_chip_id()/8;
		//ZSQ0126
		if (INTER_TOPO == 1 && (mf->get_sid()/32+mf->get_chip_id()/8)%2 == 0) //ring, forward
		    to_module = 192 + (mf->get_chip_id()/8+1)%4;
		//ZSQ0126
                unsigned size = mf->get_is_write()?mf->size():mf->get_ctrl_size();
                //printf("ZSQ: inter-module icnt_push() from %d to %d, mf is %s NULL, sid = %d chip_id = %d sub_partition_id=%u type = %d inst @ pc=0x%04x\n", from_module, to_module, mf?"not":"", mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), (int*)mf->get_type(), mf->get_pc());
                //fflush(stdout);
                if (::icnt_has_buffer(from_module, size)) {
                    ::icnt_push(from_module, to_module, (void*)mf, size);
                    m_sub_partition[spid]->L2_dram_queue_pop();
		    m_arbitration_metadata.borrow_credit(spid);
                }
            }
            else if (can_issue_to_dram(spid)){ //local. push to dram_latency_queue

                m_sub_partition[spid]->L2_dram_queue_pop();
                MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                //printf("ZSQ: sub_partition %d L2_dram_queue to dram_latency_queue, mf sid = %d chip_id = %d sub_partition_id=%u inst @ pc=0x%04x\n", spid,  mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->get_pc());
                fflush(stdout);
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                m_dram_latency_queue.push_back(d);
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.borrow_credit(spid);
                break;  // the DRAM should only accept one request per cycle
            }
        }
  }
}	

    }
#endif
ZSQ 20210130 Rearranged in the latter piece of code*/

//ZSQ 20210130 Rearranged the above piece of code here
#if SM_SIDE_LLC == 1
    std::stringstream out1;
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) { //loop sub_partition start
        int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;
        if(!KAIN_NoC_r.get_inter_icnt_pop_mem_turn(m_id)){  //L2_dram_queue turn, start
            if (!m_sub_partition[spid]->L2_dram_queue_empty()) {    //L2_dram_queue turn, !L2_dram_queue_empty, start
                mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
                if (mf->get_sid()/32 != mf->get_chip_id()/8){ //remote, push to inter_icnt
                    unsigned from_module = 192 + mf->get_sid()/32;
                    unsigned to_module = 192 + mf->get_chip_id()/8;
#if BEN_OUTPUT == 1
                    mf->set_src(from_module);
                    mf->set_dst(to_module);
                    mf->set_next_hop(to_module);
#endif
                    if (INTER_TOPO == 1 && (mf->get_sid()/32+mf->get_chip_id()/8)%2 == 0){ //ring, forward
                        to_module = 192 + (mf->get_chip_id()/8+1)%4;
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
                    mem_fetch *mf =  KAIN_NoC_r.inter_icnt_pop_mem_pop(m_id)->req;
                    if(gpu_sim_cycle >= 1000000) {
                        out1 << "icnt_mem_push_pop\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                             "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                             << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_sid()/32 << "\tsize: " << mf->size()
                             <<"\tgpu_cycle: " << gpu_sim_cycle << "\n";
                        rep3->apply(out1.str().c_str());
                    }
                    dram_delay_t d;
                    d.req = mf;
                    d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                    mf->set_icnt_cycle(d.ready_cycle);
                    m_dram_latency_queue.push_back(d);
                    dram_latency_in++;
                    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                }
            }    //L2_dram_queue turn, L2_dram_queue_empty, end
        }   //L2_dram_queue turn, end
        else {  //inter_icnt_pop_mem, start
            if (!KAIN_NoC_r.inter_icnt_pop_mem_empty(m_id)) {   //inter_icnt_pop_mem turn, !inter_icnt_pop_mem_empty, start
                mem_fetch *mf =  KAIN_NoC_r.inter_icnt_pop_mem_pop(m_id)->req;

                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                mf->set_icnt_cycle(d.ready_cycle);
                m_dram_latency_queue.push_back(d);
                if(gpu_sim_cycle >= 1000000) {
                    out1 << "icnt_mem_push_pop\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                         "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                         << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_sid()/32 << "\tsize: " << response_size
                         <<"\tgpu_cycle: " << gpu_sim_cycle << "\n";
                    rep3->apply(out1.str().c_str());
                }
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
#if BEN_OUTPUT == 1
                        mf->set_src(from_module);
                        mf->set_dst(to_module);
                        mf->set_next_hop(to_module);
#endif
                        if (INTER_TOPO == 1 && (mf->get_sid()/32+mf->get_chip_id()/8)%2 == 0) //ring, forward
                            to_module = 192 + (mf->get_chip_id()/8+1)%4;
                        unsigned size = mf->get_is_write()?mf->size():mf->get_ctrl_size();
                        if (::icnt_has_buffer(from_module, size) && m_arbitration_metadata.has_credits(spid)) {
                            ::icnt_push(from_module, to_module, (void*)mf, size);
                            if(gpu_sim_cycle >= 1000000) {
                                out1 << "dram_icnt\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                     "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                     << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_sid()/32 << "\tsize: " << mf->size()
                                     <<"\tgpu_cycle: " << gpu_sim_cycle << "\n";
                                rep3->apply(out1.str().c_str());
                            }
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
#if BEN_OUTPUT == 1
    std::ostringstream out1;
#endif
    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;
        if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid) && m_arbitration_metadata.has_credits(spid)) {
            mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
            printf("");
            m_sub_partition[spid]->L2_dram_queue_pop();
            MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
            //printf("ZSQ: sub_partition %d L2_dram_queue to dram_latency_queue, mf sid = %d chip_id = %d sub_partition_id=%u inst @ pc=0x%04x\n", spid,  mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->get_pc());
            fflush(stdout);
            dram_delay_t d;
            d.req = mf;
            d.ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle + m_config->dram_latency;
            //if (mf->get_chip_id()/8 != mf->get_sid()/32) d.ready_cycle += 32; //inter link delay, MCMGPU paper, 32 cycles
            m_dram_latency_queue.push_back(d);
            dram_latency_in++;
            mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
            m_arbitration_metadata.borrow_credit(spid);
            break;  // the DRAM should only accept one request per cycle
        }
    }
#endif

/*    int last_issued_partition = m_arbitration_metadata.last_borrower();
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel;
        if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid)) {
            mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();

            if (mf->is_write())
            {   // 1 is for write, while 0 for read
                //if ( !m_dram_r->full(1, (long)mf->kain_get_addr()) )
                if ( !KAIN_NoC_r.request_full(mf, mf->get_chip_id()/8,m_id/8))
                {
                    m_sub_partition[spid]->L2_dram_queue_pop();
                    MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                    dram_delay_t d;
                    d.req = mf;

                    std::map<new_addr_type,  new_addr_type*> ::iterator iter = KAIN_page_table.find(mf->kain_cycle&0xfffe00000);
                    assert(iter != KAIN_page_table.end());
                    unsigned long long kain_cycle = (iter->second)[1];


                    d.ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle + m_config->dram_latency + kain_cycle;
                    m_dram_latency_queue.push_back(d);
                    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                    m_arbitration_metadata.borrow_credit(spid);
                    break;  // the DRAM should only accept one request per cycle

                }
            } else
            {

                //if ( !m_dram_r->full(0, (long)mf->kain_get_addr()) )
                if ( !KAIN_NoC_r.request_full(mf, mf->get_chip_id()/8,m_id/8))
                {
                    m_sub_partition[spid]->L2_dram_queue_pop();
                    MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid);
                    dram_delay_t d;
                    d.req = mf;


                    std::map<new_addr_type,  new_addr_type*> ::iterator iter = KAIN_page_table.find(mf->kain_cycle&0xfffe00000);
                    assert(iter != KAIN_page_table.end());
                    unsigned long long kain_cycle = (iter->second)[1];

                    d.ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle + m_config->dram_latency + kain_cycle;
                    //printf("kain cycle is %lld\n", kain_cycle);

                    m_dram_latency_queue.push_back(d);
                    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                    m_arbitration_metadata.borrow_credit(spid);
                    break;  // the DRAM should only accept one request per cycle
                }
            }
        }
    }
*/

    //kain_NoC_r

    if (!m_dram_latency_queue.empty() && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle)) {
        mem_fetch *mf = m_dram_latency_queue.front().req;
        if (mf->is_write()) {
            if (!m_dram_r->full(1, (long) mf->kain_get_addr()) && !m_dram_r->r_returnq_full()) {
                m_dram_r->push(mf);
                m_dram_latency_queue.pop_front();
                dram_latency_out++;
                //printf("ZSQ: dram_latency_queue to dram_push write, mf sid = %d chip_id = %d sub_partition_id=%u inst @ pc=0x%04x\n", mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->get_pc());
                fflush(stdout);
                kain_memory_page_count[mf->get_chip_id() / 8]++;
            }
        } else {
            if (!m_dram_r->full(0, (long) mf->kain_get_addr()) && !m_dram_r->r_returnq_full()) {
                m_dram_r->push(mf);
                m_dram_latency_queue.pop_front();
                dram_latency_out++;
                //printf("ZSQ: dram_latency_queue to dram_push read, mf sid = %d chip_id = %d sub_partition_id=%u inst @ pc=0x%04x\n", mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->get_pc());
                fflush(stdout);
            }
        }
    }

/*
    if ( !m_dram_latency_queue.empty() && ( (gpu_sim_cycle + gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle ) )        {


        {
            mem_fetch* mf = m_dram_latency_queue.front().req;
            //printf("mf come from tcp %d, cache id %d, MC id %d\n", mf->get_tpc(), mf->get_sub_partition_id()-m_id*m_config->m_n_sub_partition_per_memory_channel,m_id);
            bool HBM_cache = mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == GLOBAL_ACC_W;
            //HBM_cache = false;
            ////////////////////////////////add by shiqing start
	    if (HBM_cache == false || (mf->get_tpc()/32 == mf->get_chip_id()/8)) //near
	    ////////////////////////////////add by shiqing end
	    //if(HBM_cache == false || ((mf->get_tpc()<64 && (mf->get_chip_id()/8) < 2) || (mf->get_tpc()>=64 && (mf->get_chip_id()/8) >= 2)))//Near
            {
            if(!KAIN_NoC_r.request_full(mf, mf->get_chip_id()/8,m_id/8))//push the mf to its corresponding chip
            {
                KAIN_NoC_r.request_push(mf, mf->get_chip_id()/8,m_id/8); 
                m_dram_latency_queue.pop_front();

                kain_memory_page_count[mf->get_chip_id()/8]++;
            }
            }
            else//Remote
            {
#if HBM_CACHE == 1
                //kain_cache[1048576];
                bool HBM_Cache_hit = false;
                new_addr_type addr = (mf->kain_get_addr())>>7;
                //printf("addr %0x\n", addr);
                if(kain_cache[(m_id/8)][(mf->get_addr()>>7)%8388608] == (mf->get_addr()>>7))
                    HBM_Cache_hit = true;
                else
                    HBM_Cache_hit = false;

                //(((mf->get_addr())>>16) & 0x7) --->> mf->KAIN_HBM_cache_channel
                if(HBM_Cache_hit == true && KAIN_HBM_Cache_request[(m_id/8)*8+(((mf->get_addr())>>16) & 0x7)].size()<256)
                {
                    mf->kain_HBM_Cache_hit_miss = 1;
                    mf->kain_transform_to_HBM_Cache_address();
                    KAIN_HBM_Cache_request[(m_id/8)*8+mf->kain_HBM_cache_channel].push_back(mf); 
                    m_dram_latency_queue.pop_front();
                }
                if(HBM_Cache_hit == false && KAIN_HBM_Cache_request[(m_id/8)*8+(((mf->get_addr())>>16) & 0x7)].size()<256 && KAIN_Remote_Memory_request.size() < 256)
                {
                    mf->kain_miss_HBM_cache = 1;
                    KAIN_Remote_Memory_request.push_back(mf);
                    m_dram_latency_queue.pop_front();

                    //KAIN TODO, this addrr needs to be changed, the same with the read HBM Cache addr
                    mem_access_t access( GLOBAL_ACC_R,mf->get_addr(), 128, 1);
                    mem_fetch *mf_tmp = new mem_fetch( access,
                        NULL,
                        32, // flit size 32
                        -1,
                        mf->get_sid(),
                        mf->get_tpc(),
                        mf->get_mem_config());

                    mf_tmp->kain_stream_id = mf->kain_stream_id;
                    mf_tmp->kain_HBM_Cache_hit_miss = 0;//miss

                    mf_tmp->kain_transform_to_HBM_Cache_address();
                    KAIN_HBM_Cache_request[(m_id/8)*8+mf_tmp->kain_HBM_cache_channel].push_back(mf_tmp);
                }
#endif
#if HBM_CACHE == 0
		if (KAIN_Remote_Memory_request.size() < 256)
		{
		    KAIN_Remote_Memory_request.push_back(mf);
                    m_dram_latency_queue.pop_front();
		}
#endif
            }
        }
     }

#if HBM_CACHE == 1
     if(!KAIN_HBM_Cache_request[m_id].empty())//we need to mimic the HBM Cache here
     {
        mem_fetch* mf = KAIN_HBM_Cache_request[m_id].front();
        //KAIN TODO: We need to give mf a new kain_get_addr here.
        if (mf->is_write())
        {   
            if ( !m_dram_r->full(1, (long)mf->kain_get_addr()) && !m_dram_r->r_returnq_full())
            {   
                KAIN_HBM_Cache_request[m_id].pop_front();
                m_dram_r->push(mf);
            }   
        } 
        else
        {   
            if ( !m_dram_r->full(0, (long)mf->kain_get_addr()) && !m_dram_r->r_returnq_full())
            {   
                KAIN_HBM_Cache_request[m_id].pop_front();
                m_dram_r->push(mf);
            }
        }
     }
#endif

     if(!KAIN_Remote_Memory_request.empty())//HBM Cache miss, need to access remote memory
     {
         mem_fetch* mf = KAIN_Remote_Memory_request.front();
         if(!KAIN_NoC_r.request_full(mf, mf->get_chip_id()/8,m_id/8))//push the mf to its corresponding chip
         {
            KAIN_NoC_r.request_push(mf, mf->get_chip_id()/8,m_id/8); 
            KAIN_Remote_Memory_request.pop_front();

            kain_memory_page_count[mf->get_chip_id()/8]++;
         }
     }

    if(!KAIN_NoC_r.request_empty(m_id))
    {
        mem_fetch* mf = KAIN_NoC_r.request_top(m_id);
        assert(mf !=NULL);
        if (mf->is_write())
        {
            if ( !m_dram_r->full(1, (long)mf->kain_get_addr()) && !m_dram_r->r_returnq_full())
            {
                KAIN_NoC_r.request_pop_front(m_id);
                m_dram_r->push(mf);
            }
        }
        else
        {
            if ( !m_dram_r->full(0, (long)mf->kain_get_addr()) && !m_dram_r->r_returnq_full())
            {
                KAIN_NoC_r.request_pop_front(m_id);
                m_dram_r->push(mf);
            }
        }
    }
*/
}

void memory_partition_unit::set_done(mem_fetch *mf) {
    unsigned global_spid = mf->get_sub_partition_id();
    int spid = global_sub_partition_id_to_local_id(global_spid);
    assert(m_sub_partition[spid]->get_id() == global_spid);
    assert(mf->kain_type != CONTEXT_WRITE_REQUEST);
    if (mf->get_access_type() == L1_WRBK_ACC || mf->get_access_type() == L2_WRBK_ACC ||
        mf->kain_type == CONTEXT_WRITE_REQUEST || mf->kain_type == CONTEXT_READ_REQUEST) {
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


void memory_partition_unit::print_stat(FILE *fp) const {
    m_dram_r->finish();
    //FIX ME to print the statistics data
}


void memory_partition_unit::print(FILE *fp) const {
    fprintf(fp, "Memory Partition %u: \n", m_id);
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->print(fp);
    }

/*    fprintf(fp, "In Dram Latency Queue (total = %zd): \n", m_dram_latency_queue.size()); 
    for (std::list<dram_delay_t>::const_iterator mf_dlq = m_dram_latency_queue.begin(); 
         mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
        mem_fetch *mf = mf_dlq->req; 
        fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle); 
        if (mf) 
            mf->print(fp); 
        else 
            fprintf(fp, " <NULL mem_fetch?>\n"); 
    }
*/
//    m_dram->print(fp); 
    m_dram_r->finish();
}

memory_sub_partition::memory_sub_partition(unsigned sub_partition_id,
                                           const struct memory_config *config,
                                           class memory_stats_t *stats) {
    m_id = sub_partition_id;
    m_config = config;
    m_stats = stats;

    assert(m_id < m_config->m_n_mem_sub_partition);

    char L2c_name[32];
    snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
    m_L2interface = new L2interface(this);
    m_mf_allocator = new partition_mf_allocator(config);

    if (!m_config->m_L2_config.disabled()) {
        m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1, m_L2interface, m_mf_allocator,
                                 IN_PARTITION_L2_MISS_QUEUE);
        m_L2cache->set_sub_partition_id(m_id);
    }

    unsigned int icnt_L2;
    unsigned int L2_dram;
    unsigned int dram_L2;
    unsigned int L2_icnt;
    sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram, &dram_L2, &L2_icnt);
    m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
    m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
    m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
    m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
    wb_addr = -1;
}

memory_sub_partition::~memory_sub_partition() {
    delete m_icnt_L2_queue;
    delete m_L2_dram_queue;
    delete m_dram_L2_queue;
    delete m_L2_icnt_queue;
    delete m_L2cache;
    delete m_L2interface;
}

//ZSQ data sharing record
std::map <new_addr_type, module_record> record_window; //[i*4+j]: module i's local llc accessed by module j. in mem-side-llc, i is m_id/8
std::map <new_addr_type, module_record> record_total;
//ZSQ profile
unsigned long long profile_llc[64][6];
unsigned long long profile_mem[32];
unsigned long long profile_tag[64][16];
unsigned long long profile_hit_miss[64][16][4];

extern unsigned long long llc_w;
extern unsigned long long llc_r;

//unsigned long long kain_request_number = 0;
void memory_sub_partition::cache_cycle(unsigned cycle) {
    std::ostringstream out;
    // L2 fill responses
    if (!m_config->m_L2_config.disabled()) {
        if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
            mem_fetch *mf = m_L2cache->next_access();
            if (mf->get_access_type() != L2_WR_ALLOC_R) { // Don't pass write allocate read request back to upper level cache
                mf->set_reply();
                mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
                if(gpu_sim_cycle >= 1000000) {
                    out << "L2_icnt_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                        "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                        << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize:"
                        << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tfrom L2\n";
                    rep4->apply(out.str().c_str());
                }
                m_L2_icnt_queue->push(mf);
            } else {
                m_request_tracker.erase(mf);
                delete mf;
            }
        }
    }

    // DRAM to L2 (texture) and icnt (not texture)
    if (!m_dram_L2_queue->empty()) {
        mem_fetch *mf = m_dram_L2_queue->top();
        if (mf->kain_type != CONTEXT_READ_REQUEST && !m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
            if (m_L2cache->fill_port_free()) {
                mf->set_status(IN_PARTITION_L2_FILL_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                m_L2cache->fill(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
                m_dram_L2_queue->pop();
                dram_L2_out++;
            }
        }
        else if (!m_L2_icnt_queue->full()) {
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
            unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
            if(gpu_sim_cycle >= 1000000) {
                out << "L2_icnt_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize:"
                    << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tfrom DRAM\n";
                rep4->apply(out.str().c_str());
            }
            m_L2_icnt_queue->push(mf);
            m_dram_L2_queue->pop();
            dram_L2_out++;
        }
    }

    // prior L2 misses inserted into m_L2_dram_queue here
    if (!m_config->m_L2_config.disabled())
        m_L2cache->cycle();

    // new L2 texture accesses and/or non-texture accesses
    if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
        mem_fetch *mf = m_icnt_L2_queue->top();
        if ((mf->kain_type != CONTEXT_WRITE_REQUEST && mf->kain_type != CONTEXT_READ_REQUEST) &&
            !m_config->m_L2_config.disabled() && ((m_config->m_L2_texure_only && mf->istexture()) || (!m_config->m_L2_texure_only))) {
            // L2 is enabled and access is for L2
            bool output_full = m_L2_icnt_queue->full();
            bool port_free = m_L2cache->data_port_free();
            if (!output_full && port_free) {
                std::list <cache_event> events;
                enum cache_request_status status = m_L2cache->access(mf->get_addr(), mf,
                                                                     gpu_sim_cycle + gpu_tot_sim_cycle, events);
                bool write_sent = was_write_sent(events);
                bool read_sent = was_read_sent(events);

                //ZSQ data sharing record
                std::map<unsigned long long, module_record>::iterator it;
                it = record_window.find(mf->get_addr() >> 7); //block_addr, cache line size 128B
                if (it != record_window.end()) // not first time in this time_window
                {
                    it->second.record[m_id / 16][mf->get_sid() / 32]++; //record from which module this mf comes
                    it->second.record_sm[m_id / 16][mf->get_sid()]++; //record from which module this mf comes
                    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
                } else //first time access in this window
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
                    }
                    record_window.insert(std::map<new_addr_type, module_record>::value_type(mf->get_addr() >> 7, tmp));
                    it = record_window.find(mf->get_addr() >> 7);
                    it->second.record[m_id / 16][mf->get_sid() / 32]++; //record from which module this mf comes
                    it->second.record_sm[m_id / 16][mf->get_sid()]++; //record from which module this mf comes
                    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
                }
                it = record_total.find(mf->get_addr() >> 7); //block_addr, cache line size 128B
                if (it != record_total.end()) // not first time in this time_window
                {
                    it->second.record[m_id / 16][mf->get_sid() / 32]++; //record from which module this mf comes
                    it->second.record_sm[m_id / 16][mf->get_sid()]++; //record from which module this mf comes
                    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
                } else //first time access in this window
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
                    }
                    record_total.insert(std::pair<new_addr_type, module_record>(mf->get_addr() >> 7, tmp));
                    it = record_total.find(mf->get_addr() >> 7);
                    it->second.record[m_id / 16][mf->get_sid() / 32]++; //record from which module this mf comes
                    it->second.record_sm[m_id / 16][mf->get_sid()]++; //record from which module this mf comes
                    it->second.rwtag |= mf->is_write(); //0 for read, 1 for write
                }

                if (status == HIT) {
                    if (!write_sent) {
                        // L2 cache replies
                        assert(!read_sent);
                        if (mf->get_access_type() == L1_WRBK_ACC) {
                            m_request_tracker.erase(mf);
                            delete mf;
                        } else {
                            mf->set_reply();
                            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
                            unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
                            if(gpu_sim_cycle >= 1000000) {
                                out << "L2_icnt_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                                    << "\tsize:" << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tcache hit\n";
                                rep4->apply(out.str().c_str());
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
                } else if (status != RESERVATION_FAIL) {
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
                    icnt_L2_out++;
                } else {
                    assert(!write_sent);
                    assert(!read_sent);
                    //printf("KAIN L2 reservartion fail\n");
                    //fflush(stdout);
                    // L2 cache lock-up: will try again next cycle
                }
            } else { ;
                //printf("KAIN the port is not free, output full %d, port full %d\n", output_full, port_free); 
                //fflush(stdout);
            }
        }
        else {
            // L2 is disabled or non-texture access to texture-only L2
            mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
            m_L2_dram_queue->push(mf);
            L2_dram_in++;
            unsigned request_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
            if(gpu_sim_cycle >= 1000000) {
                out << "L2_DRAM_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                    << "\tsize:" << request_size <<"\tgpu_cycle: " << gpu_sim_cycle << "\tcache miss\n";
                rep4->apply(out.str().c_str());
            }
            m_icnt_L2_queue->pop();
            icnt_L2_out++;
        }
    }
    else { ;
        //	if(m_L2_dram_queue->full() && (gpu_sim_cycle+gpu_tot_sim_cycle)%1 == 0)
        //		printf("KAIN m_L2_dram_queue is full, id %d\n", m_id);
        //	if(m_icnt_L2_queue->empty() && (gpu_sim_cycle+gpu_tot_sim_cycle)%1 == 0)
        //		printf("KAIN m_icnt_L2 queue is empty \n");
    }

    // ROP delay queue
    if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) && !m_icnt_L2_queue->full()) {
        mem_fetch *mf = m_rop.front().req;
        if (mf->is_write()) llc_w++;
        else llc_r++;
        m_rop.pop();
        rop_out++;
        unsigned request_size;
        if(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_ACK)
            request_size = 8;
        else if(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_REQUEST)
            request_size = 136;
#if 0
        /*if(gpu_sim_cycle > 1000000) {
            out << "rop pop\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize:" << request_size
                << "\n";
            rep4->apply(out.str().c_str());
        }*/
#endif
        m_icnt_L2_queue->push(mf);

        icnt_L2_in++;
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
    }
}

bool memory_sub_partition::full() const {
    return m_icnt_L2_queue->full();
}

bool memory_sub_partition::L2_dram_queue_empty() const {
    return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
    return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() {
    m_L2_dram_queue->pop();
    L2_dram_out++;
}

bool memory_sub_partition::dram_L2_queue_full() const {
    return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
    m_dram_L2_queue->push(mf);
    dram_L2_in++;
}

void memory_sub_partition::print_cache_stat(unsigned &accesses, unsigned &misses) const {
    FILE *fp = stdout;
    if (!m_config->m_L2_config.disabled())
        m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {

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
    if (!m_config->m_L2_config.disabled())
        m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
    // gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
    // gzprintf(visualizer_file, "Ltwowritehit: %d\n",  L2_write_access-L2_write_miss);
    // gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
    // gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_access-L2_read_miss);
    if (num_mfs)
        gzprintf(visualizer_file, "averagemflatency: %lld\n", mf_total_lat / num_mfs);
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
    unsigned cmd = 0;
    unsigned activity = 0;
    unsigned nop = 0;
    unsigned act = 0;
    unsigned pre = 0;
    unsigned rd = 0;
    unsigned wr = 0;
    unsigned req = 0;
    unsigned tot_cmd = 0;
    unsigned tot_nop = 0;
    unsigned tot_act = 0;
    unsigned tot_pre = 0;
    unsigned tot_rd = 0;
    unsigned tot_wr = 0;
    unsigned tot_req = 0;

    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
//		m_memory_partition_unit[i]->set_dram_power_stats(cmd,activity,nop,act,pre,rd,wr,req);
        tot_cmd += cmd;
        tot_nop += nop;
        tot_act += act;
        tot_pre += pre;
        tot_rd += rd;
        tot_wr += wr;
        tot_req += req;
    }
    fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
    fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
    fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
    fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
    fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
    fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
    fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->flush();
    }
    return 0; // L2 is read only in this version
}

bool memory_sub_partition::busy() const {
    return !m_request_tracker.empty();
}

void memory_sub_partition::push(mem_fetch *req, unsigned long long cycle) {
    if (req) {
        m_request_tracker.insert(req);
        m_stats->memlatstat_icnt2mem_pop(req);
        if (req->istexture()) {
            m_icnt_L2_queue->push(req);
            icnt_L2_in++;
            req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
        } else {
            rop_delay_t r;
            r.req = req;
            r.ready_cycle = cycle + m_config->rop_latency;
            m_rop.push(r);
            rop_in++;
            req->set_status(IN_PARTITION_ROP_DELAY, gpu_sim_cycle + gpu_tot_sim_cycle);
            req->mf_print();
        }
    }
}

mem_fetch *memory_sub_partition::pop() {
    mem_fetch *mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if (mf && mf->isatomic())
        mf->do_atomic();
    if (mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC)) {
        delete mf;
        mf = NULL;
    }
    return mf;
}

mem_fetch *memory_sub_partition::top() {
    mem_fetch *mf = m_L2_icnt_queue->top();
    if (mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC)) {
        m_L2_icnt_queue->pop();
        m_request_tracker.erase(mf);
        delete mf;
        mf = NULL;
    }
    return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
    m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(class cache_stats &l2_stats) const {
    if (!m_config->m_L2_config.disabled()) {
        l2_stats += m_L2cache->get_stats();
    }
}

void memory_sub_partition::get_L2cache_sub_stats(struct cache_sub_stats &css) const {
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->get_sub_stats(css);
    }
}

void memory_sub_partition::get_L2cache_sub_stats_kain(unsigned cluster_id, struct cache_sub_stats &css) const {
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->get_sub_stats_kain(cluster_id, css);
    }
}


void memory_sub_partition::clear_L2cache_sub_stats_kain() {
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->clear_sub_stats_kain();
    }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
    // TODO: Add visualizer stats for L2 cache
}

void
memory_sub_partition::set_mk_scheduler(MKScheduler *mk_sched) {
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->set_mk_scheduler(mk_sched);
    }
}

