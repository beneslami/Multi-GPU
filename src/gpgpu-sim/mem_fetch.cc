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

#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include "gpu-sim.h"

unsigned mem_fetch::sm_next_mf_request_uid=1;

mem_fetch::mem_fetch( const mem_access_t &access, 
                      const warp_inst_t *inst,
                      unsigned ctrl_size, 
                      unsigned wid,
                      unsigned sid, 
                      unsigned tpc, 
                      const class memory_config *config )
{
   m_request_uid = sm_next_mf_request_uid++;
   m_access = access;
   if( inst ) { 
       m_inst = *inst;
       assert( wid == m_inst.warp_id() );
   }
   m_data_size = access.get_size();
   m_ctrl_size = ctrl_size;
   m_sid = sid;
   m_tpc = tpc;
   m_wid = wid;
   config->m_address_mapping.addrdec_tlx(access.get_addr(),&m_raw_addr,this,m_tpc);
   kain_new_addr = config->m_address_mapping.kain_addrdec_tlx(access.get_addr(), this);
   kain_new_addr_back = kain_new_addr;

   m_src = 0;
   m_dst = 0;
   m_next_hop = 0;
   m_flag = false;
   m_create = 0;
   m_send = 0;
   m_receive = 0;
   m_local_llc_miss = 0; // Added by Ben
   m_local_mem_miss = 0; // Added by Ben
   m_step = 1;  // Added by Ben
   m_last_time = gpu_sim_cycle;  // Added by Ben
   m_chiplet = -1;

   kain_miss_HBM_cache = 0;
   kain_HBM_cache_channel = -1;

   m_partition_addr = config->m_address_mapping.partition_address(access.get_addr());
   m_type = m_access.is_write()?WRITE_REQUEST:READ_REQUEST;

/*
   if(sm_next_mf_request_uid%8 == 0)
   kain_new_addr = 0x00000;
   else if (sm_next_mf_request_uid % 8 == 1)
   kain_new_addr = 0x00080;
   else if (sm_next_mf_request_uid % 8 == 2)
   kain_new_addr = 0x00100;
   else if (sm_next_mf_request_uid % 8 == 3)
   kain_new_addr = 0x00180;
   else if (sm_next_mf_request_uid % 8 == 4)
   kain_new_addr = 0x00200;
   else if (sm_next_mf_request_uid % 8 == 5)
   kain_new_addr = 0x00280;
   else if (sm_next_mf_request_uid % 8 == 6)
   kain_new_addr = 0x00300;
   else if (sm_next_mf_request_uid % 8 == 7)
   kain_new_addr = 0x00380;
 */   
//   if(sm_next_mf_request_uid%8 == 0)
//    kain_new_addr = 0x00000;
//   else 
//    kain_new_addr = 0x00080;

    //kain_new_addr = kain_new_addr & 0xfff80000;
    //kain_new_addr = kain_new_addr & 0xffff0180;
    //kain_new_addr = kain_new_addr & 0x00070000;
   //kain_new_addr = 0xf0000000;
 //  m_type = READ_REQUEST;//KAIN DO NOT FORGET abstract_hsaredware.h
   //m_type = WRITE_REQUEST;//KAIN DO NOT FORGET abstract_hsaredware.h
    

   m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
   m_timestamp2 = 0;
   m_status = MEM_FETCH_INITIALIZED;
   m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
   m_mem_config = config;
   icnt_flit_size = config->icnt_flit_size;
   kain_type = NO_CONTEXT;
   kain_stream_id = 0;
}

mem_fetch::~mem_fetch()
{
    m_status = MEM_FETCH_DELETED;
}

#define MF_TUP_BEGIN(X) static const char* Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::print( FILE *fp, bool print_inst ) const
{
    if( this == NULL ) {
        fprintf(fp," <NULL mem_fetch pointer>\n");
        return;
    }
    fprintf(fp,"  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid, m_wid, m_raw_addr.chip );
    m_access.print(fp);
    if( (unsigned)m_status < NUM_MEM_REQ_STAT ) 
       fprintf(fp," status = %s (%llu), ", Status_str[m_status], m_status_change );
    else
       fprintf(fp," status = %u??? (%llu), ", m_status, m_status_change );
    if( !m_inst.empty() && print_inst ) m_inst.print(fp);
    else fprintf(fp,"\n");
}

void mem_fetch::set_status( enum mem_fetch_status status, unsigned long long cycle ) 
{
    m_status = status;
    m_status_change = cycle;
}

bool mem_fetch::isatomic() const
{
   if( m_inst.empty() ) return false;
   return m_inst.isatomic();
}

void mem_fetch::do_atomic()
{
    m_inst.do_atomic( m_access.get_warp_mask() );
}

bool mem_fetch::istexture() const
{
    if( m_inst.empty() ) return false;
    return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const
{ 
    if( m_inst.empty() ) return false;
    return (m_inst.space.get_type() == const_space) || (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem){
	unsigned sz=0;
	// If atomic, write going to memory, or read coming back from memory, size = ctrl + data. Else, only ctrl
	if( isatomic() || (simt_to_mem && get_is_write()) || !(simt_to_mem || get_is_write()) )
		sz = size();
	else
		sz = get_ctrl_size();

	return (sz/icnt_flit_size) + ( (sz % icnt_flit_size)? 1:0);
}



