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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include "addrdec.h"
#include "../abstract_hardware_model.h"
#include <bitset>

enum mf_type {
   READ_REQUEST = 0,
   WRITE_REQUEST,
   READ_REPLY, // send to shader
   WRITE_ACK
};

enum context_type{
   NO_CONTEXT = 0,
   CONTEXT_READ_REQUEST ,//add by kain
   CONTEXT_WRITE_REQUEST, // add by kain
   CONTEXT_READ_REPLY, // add by kain
   CONTEXT_WRITE_ACK//add by kain
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

class mem_fetch {
public:
    mem_fetch( const mem_access_t &access, 
               const warp_inst_t *inst,
               unsigned ctrl_size, 
               unsigned wid,
               unsigned sid, 
               unsigned tpc, 
               const class memory_config *config );
   ~mem_fetch();

   void set_status( enum mem_fetch_status status, unsigned long long cycle );
   void set_reply() 
   { 
      // assert( m_access.get_type() != L1_WRBK_ACC && m_access.get_type() != L2_WRBK_ACC );
       if( m_type==READ_REQUEST ) {
         //  assert( !get_is_write() );
           m_type = READ_REPLY;
       } else if( m_type == WRITE_REQUEST ) {
          // assert( get_is_write() );
           m_type = WRITE_ACK;
       }
   }

   void set_request()//KAIN used in HBM caching 
   { 
       assert( m_access.get_type() != L1_WRBK_ACC && m_access.get_type() != L2_WRBK_ACC );
       if( m_type==READ_REPLY) {
         //  assert( !get_is_write() );
           m_type = READ_REQUEST;
       } else if( m_type == WRITE_ACK ) {
          // assert( get_is_write() );
           m_type = WRITE_REQUEST;
       }
       kain_miss_HBM_cache = 1;
   }
   void do_atomic();

   void print( FILE *fp, bool print_inst = true ) const;

   const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
   unsigned get_data_size() const { return m_data_size; }
   void     set_data_size( unsigned size ) { m_data_size=size; }
   unsigned get_ctrl_size() const { return m_ctrl_size; }
   unsigned size() const { return m_data_size+m_ctrl_size; }
   bool is_write() {return m_access.is_write();}
   void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
   new_addr_type get_addr() const { return m_access.get_addr(); }

   new_addr_type kain_get_addr() { return kain_new_addr; }

   void kain_transform_to_HBM_Cache_address()
   {
        new_addr_type kain_bank_addr = (get_addr() >> 7)& 0x000f;
        new_addr_type kain_column_addr = (get_addr() >> 11) & 0x001f;
        new_addr_type kain_row_addr = (get_addr() >> 21) & 0x7fff;

        kain_HBM_cache_channel = (get_addr()>>16) & 0x7;

        kain_row_addr = 0x3fff-(kain_row_addr%0x7ff);//each channel 128MB, 8 channels so 1GB per GPU

        kain_new_addr = (kain_column_addr<<7) | (kain_bank_addr<<12) | (kain_row_addr<<16);
   }
   void kain_transform_back_kain_address()
   {
        kain_new_addr = kain_new_addr_back; 
   }

   void kain_set_write()
   {
       m_type = WRITE_REQUEST; 
   }

   new_addr_type get_partition_addr() const { return m_partition_addr; }
   unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
   unsigned get_chip_id() const { return m_raw_addr.chip; }
   unsigned bankID() const { return m_raw_addr.bk; }
   unsigned columnID() const { return m_raw_addr.col; }
   bool     get_is_write() const { return m_access.is_write(); }
   unsigned get_request_uid() const { return m_request_uid; }
   unsigned get_sid() const { return m_sid; }
   unsigned get_tpc() const { return m_tpc; }
   unsigned get_wid() const { return m_wid; }
   void kain_set_tpc(unsigned tpc) { m_tpc = tpc; }
   bool istexture() const;
   bool isconst() const;
   enum mf_type get_type() const { return m_type; }
   bool isatomic() const;
   bool is_remote(){ return (m_sid != m_raw_addr.chip); }

   void set_next_hop(unsigned hop) { this->m_next_hop = hop; }
   void set_src(unsigned src) { this->m_src = src; }
   void set_dst(unsigned dst) { this->m_dst = dst; }
   void set_flag() { this->m_flag = true; }
   void unset_flag() { this->m_flag = false; }
   bool get_flag() { return this->m_flag; }
   unsigned get_src() { return this->m_src; }
   unsigned get_dst() { return this->m_dst; }
   unsigned get_next_hop() { return this->m_next_hop; }
   unsigned long long get_create() { return this->m_create; }
   unsigned long long get_send() { return this->m_send; }
   unsigned long long get_receive() { return this->m_receive; }
   void set_create(unsigned long long cycle) { this->m_create = cycle; }
   void set_send(unsigned long long cycle) { this->m_send = cycle; }
   void set_receive(unsigned long long cycle){ this->m_receive = cycle; }
   void set_local_llc_miss(unsigned long long cycle) { this->m_local_llc_miss = cycle; }
   void set_local_mem_miss(unsigned long long cycle) { this->m_local_mem_miss = cycle; }
   unsigned long long get_local_llc_miss(){ return this->m_local_llc_miss; }
   unsigned long long get_local_mem_miss(){ return this->m_local_mem_miss; }
   int get_step() { return this->m_step;}
   void add_step() { this->m_step++; }
   int get_last_time() return this->m_last_time; }
   void set_last_time(unsigned long long cycle){ this->m_last_time = cycle; }

#if SM_SIDE_LLC == 0
    unsigned long long get_remote_llc_miss() {return this->m_remote_llc; } //Added by Ben
    void set_remote_llc_miss(unsigned long long cycle){ this->m_remote_llc = cycle; } //Added by Ben
#endif

   void set_return_timestamp( unsigned t ) { m_timestamp2=t; }
   void set_icnt_receive_time( unsigned t ) { m_icnt_receive_time=t; }
   unsigned get_timestamp() const { return m_timestamp; }
   unsigned get_return_timestamp() const { return m_timestamp2; }
   unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

   enum mem_access_type get_access_type() const { return m_access.get_type(); }
   const active_mask_t& get_access_warp_mask() const { return m_access.get_warp_mask(); }
   mem_access_byte_mask_t get_access_byte_mask() const { return m_access.get_byte_mask(); }

   address_type get_pc() const { return m_inst.empty()?-1:m_inst.pc; }
   const warp_inst_t &get_inst() { return m_inst; }
   enum mem_fetch_status get_status() const { return m_status; }

   const memory_config *get_mem_config(){return m_mem_config;}

   unsigned get_num_flits(bool simt_to_mem);

   enum context_type kain_type;
   new_addr_type kain_new_addr;
   new_addr_type kain_new_addr_back;

   int kain_HBM_Cache_hit_miss;//1 hit, 0 miss

   new_addr_type kain_cycle;
   new_addr_type kain_cycle2;

   unsigned kain_stream_id;
   unsigned kain_miss_HBM_cache;
   int kain_HBM_cache_channel;
private:
   // request source information
   unsigned m_request_uid;
   unsigned m_sid;
   unsigned m_tpc;
   unsigned m_wid;

   // where is this request now?
   enum mem_fetch_status m_status;
   unsigned long long m_status_change;

   // request type, address, size, mask
   mem_access_t m_access;
   unsigned m_data_size; // how much data is being written
   unsigned m_ctrl_size; // how big would all this meta data be in hardware (does not necessarily match actual size of mem_fetch)
   new_addr_type m_partition_addr; // linear physical address *within* dram partition (partition bank select bits squeezed out)
   addrdec_t m_raw_addr; // raw physical address (i.e., decoded DRAM chip-row-bank-column address)
   enum mf_type m_type;  //read/write request/reply

   unsigned m_src; // Added by Ben
   unsigned m_dst; // Added by Ben
   unsigned m_next_hop; // Added by Ben
   bool m_flag;    // Added by Ben
   unsigned long long m_create; // Added by Ben
   unsigned long long m_local_llc_miss; // Added by Ben
   unsigned long long m_local_mem_miss; // Added by Ben
   unsigned long long m_remote_mem; // Added by Ben
#if SM_SIDE_LLC == 0
   unsigned long long m_remote_llc; //Added by Ben
#endif
   unsigned long long m_send;   // Added by Ben
   unsigned long long m_receive;// Added by Ben
   int m_step;                  // Added by Ben
   unsigned long long m_last_time;  // Added by Ben


   // statistics
   unsigned m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
   unsigned m_timestamp2; // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed onto icnt to shader; only used for reads
   unsigned m_icnt_receive_time; // set to gpu_sim_cycle + interconnect_latency when fixed icnt latency mode is enabled

   // requesting instruction (put last so mem_fetch prints nicer in gdb)
   warp_inst_t m_inst;

   static unsigned sm_next_mf_request_uid;

   const class memory_config *m_mem_config;
   unsigned icnt_flit_size;
};

#endif
