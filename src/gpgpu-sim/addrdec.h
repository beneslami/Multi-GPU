// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../option_parser.h"

#ifndef ADDRDEC_H
#define ADDRDEC_H

#include "../abstract_hardware_model.h"

// added by yuxi, 3d dm part
#define MAX_ADDR_BIT 32

struct addrdec_t {
   void print( FILE *fp ) const;
    
   unsigned chip; // actually channel id; in 3d memory it means stack id; commented by freeman, 3d part
   unsigned vault; // vault id;
   unsigned layer; // layer id; for each vault has multiple layers(or dies or hmc)
   unsigned bk;
   unsigned row;
   unsigned col;
   unsigned burst;

   unsigned sub_partition; 
   unsigned sub_partition_kain; 
};

class linear_to_raw_address_translation {
public:
   linear_to_raw_address_translation();
   void addrdec_setoption(option_parser_t opp);
   void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel); 

   // accessors
   void addrdec_tlx(new_addr_type addr, addrdec_t *tlx,mem_fetch * mf, unsigned tpc) const; 
   new_addr_type kain_addrdec_tlx(new_addr_type p_addr, mem_fetch *mf) const;
   new_addr_type partition_address( new_addr_type addr ) const;

   // added by yuxi, 3d dm part
   bool la2pa_mapping_enabled;

   // la2pa has not finished yet
   void addrdec_la2pa_option(const char *option);
   new_addr_type addrdec_la2pa_tlx(new_addr_type raw_addr) const;

   // random part
   new_addr_type get_raw_addr(new_addr_type addr) const;

   bool random_mapping_enabled;
   new_addr_type addrdec_random_tlx(new_addr_type raw_addr) const;

   bool pre_part_random_mapping_enabled;
   new_addr_type addrdec_part_random_tlx(new_addr_type raw_addr) const;

   // no_row and no_col_row share the same addrdec_random_tlx
   bool part_random_mapping_enabled;
   bool part_v2_random_mapping_enabled;
   bool s1_random_mapping_enabled;
   bool s2_random_mapping_enabled;
   bool s3_random_mapping_enabled;
   bool s4_random_mapping_enabled;
   bool s1_v2_random_mapping_enabled;
   bool s2_v2_random_mapping_enabled;
   bool s3_v2_random_mapping_enabled;
   bool s4_v2_random_mapping_enabled;

   bool s1_v3_random_mapping_enabled;
   bool s3_v3_random_mapping_enabled;

   bool overall_v2_random_mapping_enabled;
   bool overall_v3_random_mapping_enabled;

   unsigned get_num_bits_per_dram_unit(new_addr_type addr);

   // row xor mapping
   bool remap_mtx_mapping_enabled;
   char * remap_matrix_option;
   new_addr_type addrdec_remap_mtx_tlx(new_addr_type raw_addr) const;

private:
   void addrdec_parseoption(const char *option);
   void sweep_test() const; // sanity check to ensure no overlapping

   // changed by freeman, 3d part
   // enum {
   //    CHIP  = 0,
   //    BK    = 1,
   //    ROW   = 2,
   //    COL   = 3,
   //    BURST = 4,
   //    N_ADDRDEC
   // };

   enum {
      CHIP  = 0,
      VAULT = 1,
      LAYER = 2,
      BK    = 3,
      ROW   = 4,
      COL   = 5,
      BURST = 6,
      N_ADDRDEC
   }; // add die section for 3d memory, 3d part

   const char *addrdec_option;
   int gpgpu_mem_address_mask;
   bool run_test; 

   // commented by yuxi, 3d dm aprt
   // chip address starting position
   int ADDR_CHIP_S;
   unsigned char addrdec_mklow[N_ADDRDEC];
   unsigned char addrdec_mkhigh[N_ADDRDEC];
   new_addr_type addrdec_mask[N_ADDRDEC];
   new_addr_type sub_partition_id_mask; 

   unsigned int gap;
   int m_n_channel;
   int m_n_sub_partition_in_channel; 

   // added by yuxi, 3d dm part
   bool xor_mapping_enabled;
   unsigned xor_upper_bit_start;

   // random matrix initialization
   
   // common part
   new_addr_type local_mask;
   new_addr_type parallel_mask;
   unsigned num_local_bits;
   unsigned num_parallel_bits;
   unsigned num_cache_line_bits;
   unsigned num_tot_bits;

   new_addr_type* rdm_mtx_opr; // shared variable for different randomizing scheme
   void mtx_init(); // common initial part
   void rdm_init();
   void part_rdm_init();
   void part_v2_rdm_init();
   void s1_rdm_init();
   void s2_rdm_init();
   void s3_rdm_init();
   void s4_rdm_init();
   void s1_v2_rdm_init();
   void s2_v2_rdm_init();
   void s3_v2_rdm_init();
   void s4_v2_rdm_init();

   void s1_v3_rdm_init();
   void s3_v3_rdm_init();

   void overall_v2_rdm_init();
   void overall_v3_rdm_init();

   // xor mapping part
   unsigned xor_upper_bit(unsigned base, new_addr_type val) const; // xor for #chips is power_of_2
   unsigned xor_lower_upper_bit(new_addr_type val) const; // xor for #chips is not power_of_2

   // remap mapping part
   void generate_remap_matrix();
   void remap_mtx_init();
   new_addr_type* remap_mtx_opr;

   // linear address to physical address option
   new_addr_type la2pa_mask; // linear address to physical address maskt
   unsigned gpgpu_most_para_bit;
   const char *la2pa_mask_option;

};

#endif
