// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
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


#include "gpu-sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zlib.h"


#include "shader.h"
//#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "visualizer.h"
#include "stats.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class  gpgpu_sim_wrapper {};
#endif

#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a,b) (((a)>(b))?(a):(b))

extern class KAIN_GPU_chiplet KAIN_NoC_r;
//ZSQ data sharing record
extern map<new_addr_type, module_record> record_window;
extern map<new_addr_type, module_record> record_total;
//ZSQ 210215
unsigned long long sm_sharing_degree[8] = {0,0,0,0,0,0,0,0}; //1,2,3-4,5-8,9-16,17-32,33-64,65-128
unsigned long long module_sharing_degree[4] = {0,0,0,0}; //1,2,3,4
unsigned long long sm_sharing_num = 0; //Accumulate at the end of each window
unsigned long long module_sharing_num = 0; //Accumulate at the end of each window
unsigned long long access_block_number = 0; //Accumulate at the end of each window
#include <algorithm>

unsigned long long llc_w = 0;
unsigned long long llc_r = 0;

bool g_interactive_debugger_enabled=false;

unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_tot_sim_cycle = 0;
unsigned long long gpu_added_latency_cycle = 0;
int core_numbers = 0;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0;
unsigned int gpu_stall_icnt2sh = 0;

/* Clock Domains */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08
#define  CHIPLET 0x10


#define MEM_LATENCY_STAT_IMPL
#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser * opp)
{


	  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
			  	  	  	  	 &g_power_config_name,"GPUWattch XML file",
	                   "gpuwattch.xml");

	   option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
	                          &g_power_simulation_enabled, "Turn on power simulator (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
	                          &g_power_per_cycle_dump, "Dump detailed power output each cycle",
	                          "0");

	   // Output Data Formats
	   option_parser_register(opp, "-power_trace_enabled", OPT_BOOL,
	                          &g_power_trace_enabled, "produce a file for the power trace (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_trace_zlevel", OPT_INT32,
	                          &g_power_trace_zlevel, "Compression level of the power trace output log (0=no comp, 9=highest)",
	                          "6");

	   option_parser_register(opp, "-steady_power_levels_enabled", OPT_BOOL,
	                          &g_steady_power_levels_enabled, "produce a file for the steady power levels (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
			   	  &gpu_steady_state_definition, "allowed deviation:number of samples",
	                 	  "8:4");

}

void memory_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &scheduler_type,
                                "0 = fifo, 1 = FR-FCFS (defaul)", "1");
    option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR, &gpgpu_L2_queue_config,
                           "i2$:$2d:d2$:$2i",
                           "8:8:8:8");

    option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                           "Use a ideal L2 cache that always hit",
                           "0");
    option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string,
                   "unified banked L2 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "64:128:8,L:B:m:N,A:16:4,4");
    option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL, &m_L2_texure_only,
                           "L2 cache used for texture only",
                           "1");
    option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
                 "number of memory modules (e.g. memory controllers) in gpu",
                 "8");
    option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32, &m_n_sub_partition_per_memory_channel,
                 "number of memory subpartition in each memory module",
                 "1");
    option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr,
                 "number of memory chips per memory controller",
                 "1");
    option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat,
                "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
                "0");
    option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32, &gpgpu_frfcfs_dram_sched_queue_size,
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32, &gpgpu_dram_return_queue_size,
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                 "default = 4 bytes (8 bytes per cycle at DDR)",
                 "4");
    option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
                 "Burst length of each DRAM request (default = 4 data bus cycle)",
                 "4");
    option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32, &data_command_freq_ratio,
                 "Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)",
                 "2");
    option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
                "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
                "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
    option_parser_register(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                     "ROP queue latency (default 85)",
                     "85");
    option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                     "DRAM latency (default 30)",
                     "30");

    m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                   "1 = post-dominator", "1");
    option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt,
                   "shader core pipeline config, i.e., {<nthread>:<warpsize>}",
                   "1024:32");
    option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &m_L1T_config.m_config_string,
                   "per-shader L1 texture cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                   "8:128:5,L:R:m:N,F:128:4,128:2");
    option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
                   "per-shader L1 constant memory cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "64:64:2,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string,
                   "shader L1 instruction cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "4:256:4,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &m_L1D_config.m_config_string,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR, &m_L1D_config.m_config_stringPrefL1,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PreShared", OPT_CSTR, &m_L1D_config.m_config_stringPrefShared,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                   "global memory access skip L1D cache (implements -Xptxas -dlcm=cg, default=no skip)",
                   "0");

    option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem,
                 "enable perfect memory mode (no cache miss)",
                 "0");
    option_parser_register(opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
                 "group of lanes that should be read/written together)",
                 "4");
    option_parser_register(opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
                 "enable clock gated reg file for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
                 "enable clock gated lanes for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers,
                 "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                 "8192");
    option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
                 "Maximum number of concurrent CTAs in shader (default 8)",
                 "8");
    option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                 "number of processing clusters",
                 "10");
    option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32, &n_simt_cores_per_cluster,
                 "number of simd cores per cluster",
                 "3");
    option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size", OPT_UINT32, &n_simt_ejection_buffer_size,
                 "number of packets in ejection buffer",
                 "8");
    option_parser_register(opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32, &ldst_unit_response_queue_size,
                 "number of response packets in ld/st unit ejection buffer",
                 "2");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_sizeDefault,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32, &gpgpu_shmem_sizePrefShared,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
                 "Number of banks in the shared memory in each shader core (default 16)",
                 "16");
    option_parser_register(opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
                 "Limit shared memory to do one broadcast per cycle (default on)",
                 "1");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32, &mem_warp_parts,
                 "Number of portions a warp is divided into for shared memory bank conflict check ",
                 "2");
    option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
                "Specify which shader core to collect the warp size distribution from",
                "-1");
    option_parser_register(opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
                "Specify which shader core to collect the warp issue distribution from",
                "0");
    option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL, &gpgpu_local_mem_map,
                "Mapping from local memory space address to simulated GPU physical address space (default = enabled)",
                "1");
    option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &gpgpu_num_reg_banks,
                "Number of register banks (default = 8)",
                "8");
    option_parser_register(opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
             "Use warp ID in mapping registers to banks (default = off)",
             "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp", OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                "number of collector units (default = 4)",
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu", OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                "number of collector units (default = 4)",
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem", OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                "number of collector units (default = 2)",
                "2");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen", OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                "number of collector units (default = 0)",
                "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                           "number of collector unit in ports (default = 0)",
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                           "number of collector unit in ports (default = 0)",
                           "0");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch,
                            "Coalescing arch (default = 13, anything else is off for now)",
                            "13");
    option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32, &gpgpu_num_sched_per_core,
                            "Number of warp schedulers per core",
                            "1");
    option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32, &gpgpu_max_insn_issue_per_warp,
                            "Max number of instructions that can be issued per warp in one cycle by scheduler",
                            "2");
    option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32, &simt_core_sim_order,
                            "Select the simulation order of cores in a cluster (0=Fix, 1=Round-Robin)",
                            "1");
    option_parser_register(opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
                            "Pipeline widths "
                            "ID_OC_SP,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_SFU,OC_EX_MEM,EX_WB",
                            "1,1,1,1,1,1,1" );
    option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32, &gpgpu_num_sp_units,
                            "Number of SP units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32, &gpgpu_num_sfu_units,
                            "Number of SF units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
                            "Number if ldst units (default=1) WARNING: not hooked up to anything",
                             "1");
    option_parser_register(opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
                                "Scheduler configuration: < lrr | gto | two_level_active > "
                                "If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization>"
                                "For complete list of prioritization values see shader.h enum scheduler_prioritization_type"
                                "Default: gto",
                                 "gto");
}

void gpgpu_sim_config::reg_options(option_parser_t opp)
{
    gpgpu_functional_sim_config::reg_options(opp);
    m_shader_config.reg_options(opp);
    m_memory_config.reg_options(opp);
    power_config::reg_options(opp);
   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt,
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt,
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");
   option_parser_register(opp, "-liveness_message_freq", OPT_INT64, &liveness_message_freq,
               "Minimum number of seconds between simulation liveness messages (0 = always print)",
               "60");
   option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL, &gpgpu_flush_l1_cache,
                "Flush L1 cache at the end of each kernel call",
                "0");
   option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL, &gpgpu_flush_l2_cache,
                   "Flush L2 cache at the end of each kernel call",
                   "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
                "Stop the simulation at deadlock (1=on (default), 0=off)",
                "1");
   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
               &gpgpu_ptx_instruction_classification,
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)",
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode,
               "Select between Performance (default) or Functional simulation (1)",
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains,
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");
   option_parser_register(opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
                          "maximum kernels that can run concurrently on GPU", "8" );
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
               "Interval between each snapshot in control flow logger",
               "0");
   option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                          &g_visualizer_enabled, "Turn on visualizer output (1=On, 0=Off)",
                          "1");
   option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                          &g_visualizer_filename, "Specifies the output log file for visualizer",
                          NULL);
   option_parser_register(opp, "-visualizer_zlevel", OPT_INT32,
                          &g_visualizer_zlevel, "Compression level of the visualizer output log (0=no comp, 9=highest)",
                          "6");
    option_parser_register(opp, "-trace_enabled", OPT_BOOL,
                          &Trace::enabled, "Turn on traces",
                          "0");
    option_parser_register(opp, "-trace_components", OPT_CSTR,
                          &Trace::config_str, "comma seperated list of traces to enable. "
                          "Complete list found in trace_streams.tup. "
                          "Default none",
                          "none");
    option_parser_register(opp, "-trace_sampling_core", OPT_INT32,
                          &Trace::sampling_core, "The core which is printed using CORE_DPRINTF. Default 0",
                          "0");
    option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                          &Trace::sampling_memory_partition, "The memory partition which is printed using MEMPART_DPRINTF. Default -1 (i.e. all)",
                          "-1");
   ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

void gpgpu_sim::launch( kernel_info_t *kinfo )
{
  unsigned cta_size = kinfo->threads_per_cta();
  if ( cta_size > m_shader_config->n_thread_per_shader ) {
    printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n", cta_size,
        m_shader_config->n_thread_per_shader );
    printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
    printf("                 modify the CUDA source to decrease the kernel block size.\n");
    abort();
  }
  unsigned n=0;
  for(n=0; n < m_running_kernels.size(); n++ ) {
    if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());

  // kernel info, max CTA per shader
  kinfo->initialize_with_num_shaders(m_config.num_shader(), MAX_CTA_PER_SHADER);

  if(m_shader_config->num_shader() != 80)
  {
 		printf("NOT 80\n");
		fflush(stdout);
//  		assert(m_shader_config->num_shader() == 80); 
  }
  for(int i = 0; i < m_shader_config->num_shader(); i++)
  {
     unsigned max_cta_per_shader = m_shader_config->max_cta(*kinfo,i);
     kinfo->set_init_max_cta_per_shader(max_cta_per_shader,i);

  }


  scheduler->add_kernel(kinfo, kinfo->get_init_max_cta_per_shader());
  kinfo->set_switching_overhead(m_config.get_context_switch_cycle(m_shader_config->get_context_size_in_bytes(kinfo)));
}

bool gpgpu_sim::can_start_kernel()
{
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() )
           return true;
   }
   return false;
}

bool gpgpu_sim::get_more_cta_left() const
{
   if (m_config.gpu_max_cta_opt != 0) {
      if( m_total_cta_launched >= m_config.gpu_max_cta_opt )
          return false;
   }
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run() )
           return true;
   }
   return false;
}

kernel_info_t *gpgpu_sim::select_kernel(unsigned sid)
{
  kernel_info_t* kernel = scheduler->next_thread_block_to_schedule();
  //kernel_info_t* kernel = scheduler->kain_next_thread_block_to_schedule(sid);
  // TODO
  // does scheduler always have to give kernel which has more ctas to run?
  // then whenever a kernel finishes, another kernel may be waiting?
  if (kernel && !kernel->no_more_ctas_to_run()) {
    // record this kernel for stat print
    m_executing_kernels.insert(kernel);
    //scheduler->inc_SM_for_kernel(kernel, sid);
    return kernel;
  }

  return NULL;
}

unsigned gpgpu_sim::finished_kernel()
{
    if( m_finished_kernel.empty() )
        return 0;
    unsigned result = m_finished_kernel.front();
    m_finished_kernel.pop_front();
    return result;
}

void gpgpu_sim::set_kernel_done( kernel_info_t *kernel )
{
  // remove_kernel will be called when it is actually removed completely
  //scheduler->remove_kernel(kernel);

  unsigned uid = kernel->get_uid();
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t*>::iterator k;
  for( k=m_running_kernels.begin(); k!=m_running_kernels.end(); k++ ) {
    if( *k == kernel ) {
      *k = NULL;
      break;
    }
  }
  assert( k != m_running_kernels.end() );
}

void set_ptx_warp_size(const struct core_config * warp_size);

gpgpu_sim::gpgpu_sim( const gpgpu_sim_config &config )
    : gpgpu_t(config), m_config(config)
{
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;
    set_ptx_warp_size(m_shader_config);
    ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
        m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,config.g_power_config_name);
#endif

    m_shader_stats = new shader_core_stats(m_shader_config);
    m_memory_stats = new memory_stats_t(m_config.num_shader(),m_shader_config,m_memory_config);
    average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
    active_sms=(float *)malloc(sizeof(float));
    m_power_stats = new power_stat_t(m_shader_config,average_pipeline_duty_cycle,active_sms,m_shader_stats,m_memory_config,m_memory_stats);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    gpu_deadlock = false;


    m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
	core_numbers = m_shader_config->n_simt_clusters;
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++)
        m_cluster[i] = new simt_core_cluster(this,i,m_shader_config,m_memory_config,m_shader_stats,m_memory_stats);

    m_memory_partition_unit = new memory_partition_unit*[m_memory_config->m_n_mem];
    m_memory_sub_partition = new memory_sub_partition*[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
        m_memory_partition_unit[i] = new memory_partition_unit(i, m_memory_config, m_memory_stats);
        for (unsigned p = 0; p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
            unsigned submpid = i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
            m_memory_sub_partition[submpid] = m_memory_partition_unit[i]->get_sub_partition(p);
        }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,m_memory_config->m_n_mem_sub_partition);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout, "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize( config.max_concurrent_kernel, NULL );
    m_last_issued_kernel = 0;
    m_last_cluster_issue = 0;
    *average_pipeline_duty_cycle=0;
    *active_sms=0;

    last_liveness_message_time = 0;

    scheduler = NULL;

    //ZSQ 20201208
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++ ) {
            last_window_accesses_from_to[i][j] = 0;
            last_window_misses_from_to[i][j] = 0;
        }
        for (int j = 0; j < 64; j++ ) {
            last_window_accesses_from_l2_to[j][i] = 0;
            last_window_misses_from_l2_to[j][i] = 0;
        }
    }
}

int gpgpu_sim::shared_mem_size() const
{
   return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const
{
   return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const
{
   return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const
{
   return m_config.core_freq/1000;
}

void gpgpu_sim::set_prop( cudaDeviceProp *prop )
{
   m_cuda_properties = prop;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const
{
   return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const
{
   return m_shader_config->model;
}

double kain_dram_frequency;
void gpgpu_sim_config::init_clock_domains(void )
{
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf",
          &core_freq, &icnt_freq, &l2_freq, &dram_freq);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   l2_period = 1/l2_freq;

   kain_dram_frequency = dram_freq;
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
}

void gpgpu_sim::reinit_clock_domains(void)
{
   core_time = 0;
   dram_time = 0;
   icnt_time = 0;
   l2_time = 0;
   chiplet_time = 0;
}

bool gpgpu_sim::active()
{
    if (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
       return false;
    if (m_config.gpu_max_insn_opt && (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
       return false;
    if (m_config.gpu_max_cta_opt && (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt) )
       return false;
    if (m_config.gpu_deadlock_detect && gpu_deadlock)
       return false;
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++)
       if( m_cluster[i]->get_not_completed()>0 )
           return true;;
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
       if( m_memory_partition_unit[i]->busy()>0 )
           return true;;
    if( icnt_busy() )
        return true;
    if( get_more_cta_left() )
        return true;
    return false;
}

void gpgpu_sim::init()
{
    // run a CUDA grid on the GPU microarchitecture simulator
    gpu_sim_cycle = 0;
    gpu_sim_insn = 0;
    last_gpu_sim_insn = 0;
    m_total_cta_launched=0;

    reinit_clock_domains();
    set_param_gpgpu_num_shaders(m_config.num_shader());
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++)
       m_cluster[i]->reinit();
    m_shader_stats->new_grid();
    // initialize the control-flow, memory access, memory latency logger
    if (m_config.g_visualizer_enabled) {
        create_thread_CFlogger( m_config.num_shader(), m_shader_config->n_thread_per_shader, 0, m_config.gpgpu_cflog_interval );
    }
    shader_CTA_count_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
    if (m_config.gpgpu_cflog_interval != 0) {
       insn_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size );
       shader_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size, m_config.gpgpu_cflog_interval);
       shader_mem_acc_create( m_config.num_shader(), m_memory_config->m_n_mem, 4, m_config.gpgpu_cflog_interval);
       shader_mem_lat_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
       shader_cache_access_create( m_config.num_shader(), 3, m_config.gpgpu_cflog_interval);
       set_spill_interval (m_config.gpgpu_cflog_interval * 40);
    }

    if (g_network_mode)
       icnt_init();

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
    if(m_config.g_power_simulation_enabled){
        init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,  gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif
}

void gpgpu_sim::update_stats() {
    m_memory_stats->memlatstat_lat_pw();
    gpu_tot_sim_cycle += gpu_sim_cycle;
    gpu_tot_sim_insn += gpu_sim_insn;
}

void gpgpu_sim::print_stats()
{
    ptx_file_line_stats_write_file();
    gpu_print_stat();

    if (g_network_mode) {
        printf("----------------------------Interconnect-DETAILS--------------------------------\n" );
        icnt_display_stats();
        icnt_display_overall_stats();
        printf("----------------------------END-of-Interconnect-DETAILS-------------------------\n" );
    }
}

void gpgpu_sim::deadlock_check()
{
   if (m_config.gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core %u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
             gpu_sim_insn_last_update_sid,
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update ));
      unsigned num_cores=0;
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         unsigned not_completed = m_cluster[i]->get_not_completed();
         if( not_completed ) {
             if ( !num_cores )  {
                 printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing instructions [core(# threads)]:\n" );
                 printf("GPGPU-Sim uArch: DEADLOCK  ");
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores < 8 ) {
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores >= 8 ) {
                 printf(" + others ... ");
             }
             num_cores+=m_shader_config->n_simt_cores_per_cluster;
         }
      }
      printf("\n");
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         bool busy = m_memory_partition_unit[i]->busy();
         if( busy )
             printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i );
      }
      if( icnt_busy() ) {
         printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
         icnt_display_state( stdout );
      }

      for (int i = 0; i < 32; i++) {
	 m_memory_partition_unit[i]->print(stdout);
      }
      for (int i = 0; i < 4; i++) {
	 if (KAIN_NoC_r.forward_waiting_size(i)>0)
	     printf("forward_waiting[%d] is not empty, size = %d\n", i, KAIN_NoC_r.forward_waiting_size(i));
      }
      for (int i = 0; i < 32; i++) {
         if (KAIN_NoC_r.inter_icnt_pop_mem_size(i)>0)
             printf("inter_icnt_pop_mem[%d] is not empty, size = %d\n", i, KAIN_NoC_r.inter_icnt_pop_mem_size(i));
      }
      for (int i = 0; i < 64; i++) {
         if (KAIN_NoC_r.inter_icnt_pop_llc_size(i)>0)
             printf("inter_icnt_pop_llc[%d] is not empty, size = %d\n", i, KAIN_NoC_r.inter_icnt_pop_llc_size(i));
      }
      for (int i = 0; i < 128; i++) {
         if (KAIN_NoC_r.inter_icnt_pop_sm_size(i)>0)
             printf("inter_icnt_pop_sm[%d] is not empty, size = %d\n", i, KAIN_NoC_r.inter_icnt_pop_sm_size(i));
      }

      printf("\nRe-run the simulator in gdb and use debug routines in .gdbinit to debug this\n");
      gpu_print_stat();
      fflush(stdout);
      abort();
   }
}

/// printing the names and uids of a set of executed kernels (usually there is only one)
std::string gpgpu_sim::executed_kernel_info_string()
{
  std::stringstream statout;

  statout << "KERNEL FINISHED!!" << std::endl;
  for (std::set<kernel_info_t*>::iterator it = m_executing_kernels.begin(), it_end = m_executing_kernels.end();
       it != it_end; ++it) {
    statout << "kernel_name = " << (*it)->name() << std::endl;
    statout << "kernel_launch_uid = " << (*it)->get_uid() << std::endl;
    statout << "kernel_thread_blocks = " << (*it)->executed_blocks() << "/" << (*it)->num_blocks() << std::endl;
    statout << "kernel_simulated_insts = " << (*it)->get_num_simulated_insts() << std::endl;
    statout << "kernel_has_atomic = " << (*it)->has_atomic() << std::endl;
    statout << "kernel_overwrites_input = " << (*it)->overwrites_input() << std::endl;
  }

  return statout.str();
}

void gpgpu_sim::set_cache_config(std::string kernel_name,  FuncCache cacheConfig )
{
	m_special_cache_config[kernel_name]=cacheConfig ;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
		    std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return iter->second;
			}
	}
	return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
	    	std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return true;
			}
	}
	return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name)
{
	if(has_special_cache_config(kernel_name)){
		change_cache_config(get_cache_config(kernel_name), kernel_name);
	}else{
		change_cache_config(FuncCachePreferNone, kernel_name);
	}
}

void gpgpu_sim::change_cache_config(FuncCache cache_config, std::string kernel_name)
{
	if(cache_config != m_shader_config->m_L1D_config.get_cache_status()){
		printf("FLUSH L1 Cache at configuration change between kernels\n");
		for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         //   std::cout<<m_cluster[i]->get_core(0)->get_kernel()->name()<<"  XX  " << kernel_name << std::endl;
            if(m_cluster[i]->get_core(0)->get_kernel() == NULL)
            {
                printf("cluster %d, kernel NULL\n",i);
                fflush(stdout);
			    m_cluster[i]->cache_flush();
            }
	    }
	}

	switch(cache_config){
	case FuncCachePreferNone:
		m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
		m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		break;
	case FuncCachePreferL1:
		if((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) || (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;

		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefL1, FuncCachePreferL1);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefL1;
		}
		break;
	case FuncCachePreferShared:
		if((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) || (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefShared, FuncCachePreferShared);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefShared;
		}
		break;
	default:
		break;
	}
}

void gpgpu_sim::clear_executed_kernel_info(kernel_info_t* kernel)
{
  m_executing_kernels.erase(kernel);
}

void gpgpu_sim::clear_executed_kernel_info()
{
  std::set<kernel_info_t*>::iterator it = m_executing_kernels.begin();
  std::set<kernel_info_t*>::iterator it_end = m_executing_kernels.end();
  while (it != it_end) {
    if ((*it)->no_more_ctas_to_run()) {
      std::set<kernel_info_t*>::iterator to_erase = it++;
      m_executing_kernels.erase(to_erase);
    } else {
      ++it;
    }
  }
}

std::vector<KAIN_Power_Gate_Number_Cycles> Power_gating_number_cycles;

int kain_request_flit;
int kain_reply_flit;

std::vector<int> KAIN_mem_queue_app1;
std::vector<int> KAIN_mem_queue_app2;
int KAIN_mem_sample_count= 0;

extern unsigned long long kain_all_cycles_app1;
extern unsigned long long kain_all_cycles_app2;

extern unsigned long long kain_all_mem_cycles_app1;
extern unsigned long long kain_all_mem_cycles_app2;
extern unsigned long long kain_all_com_cycles_app1;
extern unsigned long long kain_all_com_cycles_app2;

extern unsigned long long kain_stall_cycles_app1;
extern unsigned long long kain_stall_cycles_app2;

extern unsigned long long kain_warp_all_count_app1;
extern unsigned long long kain_warp_all_count_app2;

extern unsigned long long kain_warp_all_available_count_app1;
extern unsigned long long kain_warp_all_available_count_app2;

extern unsigned long long kain_warp_mem_stall_count_app1;
extern unsigned long long kain_warp_mem_stall_count_app2;

extern unsigned long long kain_warp_unit_stall_count_app1;
extern unsigned long long kain_warp_unit_stall_count_app2;

extern unsigned long long KAIN_kernel1_LLC_access;
extern unsigned long long KAIN_kernel1_LLC_hit;
extern unsigned long long KAIN_kernel2_LLC_access;
extern unsigned long long KAIN_kernel2_LLC_hit;

extern bool Stream1_SM[384];
extern bool Stream2_SM[192];

unsigned long long kain_row_hit_app1 = 0;
unsigned long long kain_row_hit_app2 = 0;
unsigned long long kain_row_miss_app1 = 0;
unsigned long long kain_row_miss_app2 = 0;

extern unsigned long long App1_write_hit;
extern unsigned long long App2_write_hit;

unsigned long long kain_cycles_HBM_app1 = 0;
unsigned long long kain_cycles_HBM_app2 = 0;
unsigned long long kain_cycles_HBM_total= 0;
unsigned long long kain_write_back_cycles = 0;

unsigned long long KAIN_request_Near;
unsigned long long KAIN_request_Remote;
unsigned long long KAIN_reply_Near;
unsigned long long KAIN_reply_Remote;

unsigned long long rop_in;
unsigned long long rop_out;
unsigned long long icnt_L2_in;
unsigned long long icnt_L2_out;
unsigned long long L2_dram_in;
unsigned long long L2_dram_out;
unsigned long long dram_latency_in;
unsigned long long dram_latency_out;
unsigned long long returnq_in;
unsigned long long returnq_out;
unsigned long long returnq_out_delete;
unsigned long long returnq_out_local;
unsigned long long returnq_out_inter;
unsigned long long returnq_out_inter_pop;
unsigned long long returnq_out_inter_pop_delete;
unsigned long long dram_L2_in;
unsigned long long dram_L2_out;
unsigned long long icnt_pop_inter;
unsigned long long icnt_pop_inter_llc;
unsigned long long icnt_pop_inter_mem;

extern int kain_memory_page_count[4];
extern long long kain_memory_page_create_count[4];
void gpgpu_sim::gpu_print_stat()
{
   FILE *statfout = stdout;

   std::string kernel_info_str = executed_kernel_info_string();
   fprintf(statfout, "%s", kernel_info_str.c_str());

   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle+gpu_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn+gpu_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn+gpu_sim_insn) / (gpu_tot_sim_cycle+gpu_sim_cycle));
   //printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);
   printf("m_total_cta_launched = %lld\n", m_total_cta_launched);
   printf("m_last_cluster_issue = %lld\n", m_last_cluster_issue);
   printf("kain_request_flit = %d\n", kain_request_flit);
   printf("kain_reply_flit = %d\n", kain_reply_flit);

   float kain_time = (gpu_tot_sim_cycle+gpu_sim_cycle)/6000000000.0;
   printf("Request_Near = %lld, Request_Near_BW = %12.4f\n", KAIN_request_Near, KAIN_request_Near*128.0/1000000000.0/kain_time);
   printf("Request_Remote = %lld, Request_Remote_BW = %12.4f\n", KAIN_request_Remote, KAIN_request_Remote*128.0/1000000000.0/kain_time);
   printf("Reply_Near = %lld, Reply_Near_BW = %12.4f\n", KAIN_reply_Near, KAIN_reply_Near*128.0/1000000000.0/kain_time);
   printf("Reply_Remote = %lld, Reply_Remote_BW = %12.4f\n", KAIN_reply_Remote, KAIN_reply_Remote*128.0/1000000000.0/kain_time);

   printf(" rop_in = %lld, rop_out = %lld\n", rop_in, rop_out);
   printf(" icnt_L2_in = %lld, icnt_L2_out = %lld\n", icnt_L2_in, icnt_L2_out);
   printf(" L2_dram_in = %lld, L2_dram_out = %lld\n", L2_dram_in, L2_dram_out);
   printf(" dram_latency_in = %lld, dram_latency_out = %lld\n", dram_latency_in, dram_latency_out);
   printf(" returnq_in = %lld, returnq_out = %lld, returnq_out_delete = %lld\n", returnq_in, returnq_out, returnq_out_delete);
   printf(" returnq_out_local = %lld, returnq_out_inter = %lld, returnq_out_inter_pop = %lld, returnq_out_inter_pop_delete = %lld\n", returnq_out_local, returnq_out_inter, returnq_out_inter_pop, returnq_out_inter_pop_delete);
   printf(" icnt_pop_inter = %lld, icnt_pop_inter_llc = %lld, icnt_pop_inter_mem = %lld\n", icnt_pop_inter, icnt_pop_inter_llc, icnt_pop_inter_mem);
   printf(" dram_L2_in = %lld, dram_L2_out = %lld\n", dram_L2_in, dram_L2_out);

    for(int i = 0; i < 4;i ++)
        printf("Stack %d, page access count %d\n", i,  kain_memory_page_count[i]);

    for(int i = 0; i < 4;i ++)
        printf("Stack %d, page count %lld\n", i,  kain_memory_page_create_count[0]/4);
//   printf("app1 All Issue Cycles = %lld, Stall Cycles = %lld, Warp All count = %lld, Warp All Available count = %lld, Warp Mem Stall count = %lld, Warp Unit Stall count = %lld\n", kain_all_cycles_app1, kain_stall_cycles_app1, kain_warp_all_count_app1, kain_warp_all_available_count_app1, kain_warp_mem_stall_count_app1, kain_warp_unit_stall_count_app1);
//   printf("app2 All Issue Cycles = %lld, Stall Cycles = %lld, Warp All count = %lld, Warp All Available count = %lld, Warp Mem Stall count = %lld, Warp Unit Stall count = %lld\n", kain_all_cycles_app2, kain_stall_cycles_app2, kain_warp_all_count_app2, kain_warp_all_available_count_app2, kain_warp_mem_stall_count_app2, kain_warp_unit_stall_count_app2);
//   printf("app1 MEM Issue Cycles = %lld, COM Issue Cycles = %lld\n",kain_all_mem_cycles_app1, kain_all_com_cycles_app1);
//   printf("app2 MEM Issue Cycles = %lld, COM Issue Cycles = %lld\n",kain_all_mem_cycles_app2, kain_all_com_cycles_app2);


   // performance counter for stalls due to congestion.
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh );

   time_t curr_time;
   time(&curr_time);
   unsigned long long elapsed_time = MAX( curr_time - g_simulation_starttime, 1 );
   printf( "gpu_total_sim_rate=%u\n", (unsigned)( ( gpu_tot_sim_insn + gpu_sim_insn ) / elapsed_time ) );

   extern long long KAIN_HBM_Cache_hit;
   extern long long KAIN_HBM_Cache_miss;
   printf("HBM Cache hit %lld, HBM Cache miss %lld\n", KAIN_HBM_Cache_hit,KAIN_HBM_Cache_miss);


   //shader_print_l1_miss_stat( stdout );
   shader_print_cache_stats(stdout);

   cache_stats core_cache_stats;
   core_cache_stats.clear();
   for(unsigned i=0; i<m_config.num_cluster(); i++){
       m_cluster[i]->get_cache_stats(core_cache_stats);
   }
   printf("\nTotal_core_cache_stats:\n");
   core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
   shader_print_scheduler_stat( stdout, false );

   m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
       printf("KAIN removed this support from GCC5.0\n");
       printf("KAIN you can enable it if your GCC version is less than 5.0\n");
       assert(0);
//	   m_gpgpusim_wrapper->print_power_kernel_stats(gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, kernel_info_str, true );
//	   mcpat_reset_perf_count(m_gpgpusim_wrapper);
   }
#endif

   // performance counter that are not local to one shader
   m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,m_memory_config->nbk);
   for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
      m_memory_partition_unit[i]->print(stdout);

   // L2 cache stats
   if(!m_memory_config->m_L2_config.disabled()){
       cache_stats l2_stats;
       struct cache_sub_stats l2_css;
       struct cache_sub_stats total_l2_css;
       l2_stats.clear();
       l2_css.clear();
       total_l2_css.clear();


       struct cache_sub_stats l2_css_app1;
       struct cache_sub_stats total_l2_css_app1;
       l2_css_app1.clear();
       total_l2_css_app1.clear();

       struct cache_sub_stats l2_css_app2;
       struct cache_sub_stats total_l2_css_app2;
       l2_css_app2.clear();
       total_l2_css_app2.clear();


       float locality_app1 = (float)kain_row_hit_app1/(float)(kain_row_hit_app1+kain_row_miss_app1);
       float locality_app2 = (float)kain_row_hit_app2/(float)(kain_row_hit_app2+kain_row_miss_app2);


       printf("KAIN App1 hit rate %lf\n", locality_app1);
       printf("KAIN App2 hit rate %lf\n", locality_app2);

       float bw_app1 = 0.0;
       float bw_app2 = 0.0;

       if((App1_write_hit+App2_write_hit) != 0)
       {
                bw_app1 += (float)kain_cycles_HBM_app1/(float)kain_cycles_HBM_total + (float)(App1_write_hit)/(float)(App1_write_hit+App2_write_hit)*(float)(kain_write_back_cycles)/(float)kain_cycles_HBM_total;
                bw_app2 += (float)kain_cycles_HBM_app2/(float)kain_cycles_HBM_total + (float)(App2_write_hit)/(float)(App1_write_hit+App2_write_hit)*(float)(kain_write_back_cycles)/(float)kain_cycles_HBM_total;
       }
       else
       {
                bw_app1 += (float)kain_cycles_HBM_app1/(float)kain_cycles_HBM_total;
                bw_app2 += (float)kain_cycles_HBM_app2/(float)kain_cycles_HBM_total;
       }

       printf("KAIN App1 bw utilizaiton %lf\n", bw_app1);
       printf("KAIN App2 bw utilization %lf\n", bw_app2);
#if REMOTE_CACHE == 1
	//ZSQ L1.5
	KAIN_NoC_r.remote_cache_print();
#endif

        //ZSQ data sharing record
	fprintf( stdout, "\n========= data sharing record =========\n");
        unsigned long long access_block_num[4] = {0,0,0,0}; //block in module i accessed in this time window
        unsigned long long shared_block_num[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}; //[i][j]: block in module i shared by j modules in this time window
        unsigned long long access_block = 0; //total number of accessed block in this time window
        unsigned long long shared_block = 0; //total number of shared block in this time window
        unsigned long long shared_block_n[4] = {0,0,0,0}; //[i]:block in module i shared in this time window
        unsigned long long shared_block_intra_module[4] = {0,0,0,0}; //[i]:block in module i shared in this time window
	unsigned long long total_access_write = 0;
        unsigned long long total_shared_module_write = 0;
/*        for (std::map<new_addr_type, module_record>::iterator it = record_total.begin(), it_end = record_total.end(); it != it_end; ++it) {
            access_block++;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
		  if (it->second.record[i][j]>1) {
		    shared_block_intra_module[j]++; // SMs in module j share this block 
		  }
                    int shared_by = (it->second.record[i][j]>0)?1:0+(it->second.record[i][j]>0)?1:0+(it->second.record[i][j]>0)?1:0+(it->second.record[i][j]>0)?1:0;
                    if (shared_by > 0) {
                        access_block_num[i]++;
                        shared_block_num[i][shared_by-1]++;
                        if (shared_by > 1) {
                            shared_block_n[i]++;
                            shared_block++;
                        }
                        break;
                    }
                }
            }
        }
*/
	//ZSQ 210215
        for (std::map<new_addr_type, module_record>::iterator it = record_total.begin(), it_end = record_total.end(); it != it_end; ++it) {
            access_block++;
	    if(it->second.rwtag) total_access_write++; //20210403
            bool it_done = false;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (it->second.record[i][j]>0) { //this block is in MEM i, record[i][j=0-4]
                        access_block_num[i]++;

                        int shared_by = ((it->second.record[i][0]>0)?1:0)+((it->second.record[i][1]>0)?1:0)+((it->second.record[i][2]>0)?1:0)+((it->second.record[i][3]>0)?1:0); //total module num that access this block in this window
                        shared_block_num[i][shared_by-1]++;
                        if (shared_by > 1) { //sharing cross modules
                            shared_block_n[i]++;
                            shared_block++;
			    if(it->second.rwtag) total_shared_module_write++; //20210403
                        }

                        it_done = true;
                        break;
                    }
                }
                if (it_done) break;
            }
        }
	//fprintf( stdout, "SMs in the same module sharing: %llu blocks for module 0, %llu blocks for module 1, %llu blocks for module 2, %llu blocks for module 3\n", shared_block_intra_module[0], shared_block_intra_module[1], shared_block_intra_module[2], shared_block_intra_module[3]);
        for (int i = 0; i < 4; i++) {
            if (access_block_num[i] == 0) fprintf( stdout, "Module MEM %d: access 0 block\n", i);
            else fprintf( stdout, "Module MEM %d: access %llu blocks, shared %llu blocks (%.4lf): %lld shared by 2 (%.4lf), %lld shared by 3 (%.4lf), %lld shared by 4 (%.4lf)\n", i, access_block_num[i], shared_block_n[i], (double)shared_block_n[i]/(double)access_block_num[i], shared_block_num[i][1], (double)shared_block_num[i][1]/(double)access_block_num[i], shared_block_num[i][2], (double)shared_block_num[i][2]/(double)access_block_num[i], shared_block_num[i][3], (double)shared_block_num[i][3]/(double)access_block_num[i]);
        }
        if (access_block == 0) fprintf( stdout, "Total: access 0 block\n");
        else fprintf( stdout, "Total: access %llu blocks, shared %llu blocks (%.4lf)\n", access_block, shared_block, (double)shared_block/(double)access_block);
	//20210403
        fprintf( stdout, "Write rate: %.4lf blocks shared by modules is write, %.4lf blocks accessed is write\n", (double)total_shared_module_write/(double)shared_block, (double)total_access_write/(double)access_block);
        fprintf( stdout, "\n");

       printf("\n========= L2 cache stats =========\n");
       for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++){
           m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
           m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

           fprintf( stdout, "L2_cache_bank[%d]: Access = %u, Miss = %u, Miss_rate = %.3lf, Pending_hits = %u, Reservation_fails = %u\n",
                    i, l2_css.accesses, l2_css.misses, (double)l2_css.misses / (double)l2_css.accesses, l2_css.pending_hits, l2_css.res_fails);

           total_l2_css += l2_css;


            for(unsigned j = 0; j < 80; j++)
            {
                if(Stream1_SM[j] == true)
                {
                    m_memory_sub_partition[i]->get_L2cache_sub_stats_kain(j,l2_css_app1);
                    total_l2_css_app1 += l2_css_app1;
                }
            }
            for(unsigned j = 0; j < 80; j++)
            {
                if(Stream2_SM[j] == true)
                {
                    m_memory_sub_partition[i]->get_L2cache_sub_stats_kain(j,l2_css_app2);
                    total_l2_css_app2 += l2_css_app2;
                }
            }
       }

        //ZSQ 20201117
        struct cache_sub_stats total_css_tmp;
        total_css_tmp.clear();
	long long total_access[3]; //0:local; 1:near; 2:remote
	long long total_miss[3]; //0:local; 1:near; 2:remote
	for (int i = 0; i < 3; i++) {
	    total_access[i] = 0;
	    total_miss[i] = 0;
	}
        for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++){
            m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);
            bool has_access = false;
            for (int j = 0; j < 4; j++) {
                if (l2_css.accesses_to[j] != 0) //ZSQ: mem_side L2 bank accesses_to[j] is accesses from chiplet j
                    has_access = true;
            }
            if (has_access) {
                fprintf( stdout, "\tL2_cache_bank[%d]:\n", i);
                for (int j = 0; j < 4; j++) {
                    if (l2_css.accesses_to[j] != 0) {
#if SM_SIDE_LLC == 1
                        fprintf( stdout, "  Access_to[%d] = %d, Miss_to[%d] = %d, Miss_rate_to[%d] = %.3lf, Pending_hits_to[%d] = %u, Reservation_fails_to[%d] = %u\n",
                         j, l2_css.accesses_to[j], j, l2_css.misses_to[j], j, (double)l2_css.misses_to[j] / (double)l2_css.accesses_to[j], j, l2_css.pending_hits_to[j], j, l2_css.res_fails_to[j]);
#endif
#if SM_SIDE_LLC == 0
                        fprintf( stdout, "  Access_from[%d] = %d, Miss_from[%d] = %d, Miss_rate_from[%d] = %.3lf, Pending_hits_from[%d] = %u, Reservation_fails_from[%d] = %u\n",
                         j, l2_css.accesses_to[j], j, l2_css.misses_to[j], j, (double)l2_css.misses_to[j] / (double)l2_css.accesses_to[j], j, l2_css.pending_hits_to[j], j, l2_css.res_fails_to[j]);
#endif
            	    }
		}
	    }
            total_css_tmp += l2_css;
            if  ((i+1)%16 == 0) {
#if SM_SIDE_LLC == 1
                fprintf( stdout, "  From chiplet%d to chiplets:\n", i/16);
#endif
#if SM_SIDE_LLC == 0
		fprintf( stdout, "  From chiplets to chiplet%d:\n", i/16);
#endif
                for (int j = 0; j < 4; j++) {
		    if (i/16 == j) { //local
			total_access[0] += total_css_tmp.accesses_to[j];
			total_miss[0] += total_css_tmp.misses_to[j];
		    } else if ((i/16+j)%2) { //near
			total_access[1] += total_css_tmp.accesses_to[j];
			total_miss[1] += total_css_tmp.misses_to[j];
		    } else {
			total_access[2] += total_css_tmp.accesses_to[j];
                        total_miss[2] += total_css_tmp.misses_to[j];
		    }
#if SM_SIDE_LLC == 1
                    fprintf( stdout, "      Access_from_to[%d][%d] = %d, Miss_from_to[%d][%d] = %d, Miss_rate_to[%d][%d] = %.3lf, Pending_hits_from_to[%d][%d] = %u, Reservation_fails_from_to[%d][%d] = %u\n",
                     i/16, j, total_css_tmp.accesses_to[j], i/16, j, total_css_tmp.misses_to[j], i/16, j, (double)total_css_tmp.misses_to[j] / (double)total_css_tmp.accesses_to[j],  i/16, j, total_css_tmp.pending_hits_to[j], i/16, j, total_css_tmp.res_fails_to[j]);
#endif
#if SM_SIDE_LLC == 0
                    fprintf( stdout, "      Access_from_to[%d][%d] = %d, Miss_from_to[%d][%d] = %d, Miss_rate_to[%d][%d] = %.3lf, Pending_hits_from_to[%d][%d] = %u, Reservation_fails_from_to[%d][%d] = %u\n",
                     j, i/16, total_css_tmp.accesses_to[j], j, i/16, total_css_tmp.misses_to[j], j, i/16, (double)total_css_tmp.misses_to[j] / (double)total_css_tmp.accesses_to[j],  j, i/16, total_css_tmp.pending_hits_to[j], j, i/16, total_css_tmp.res_fails_to[j]);

#endif
		}
                total_css_tmp.clear();
            }
        }
/*        for (int i = 0; i < 4; i++) {
            fprintf( stdout, "\tL2_total_cache_accesses_from[%d] = %u\n", i, total_l2_css.accesses_to[i]);
            fprintf( stdout, "\tL2_total_cache_misses_from[%d] = %u\n", i, total_l2_css.misses_to[i]);
            if(total_l2_css.accesses_to[i] > 0){
                fprintf( stdout, "\tL2_total_cache_miss_rate_from[%d] = %.4lf\n", i, (double)total_l2_css.misses_to[i] / (double)total_l2_css.accesses_to[i]);
            }
            fprintf( stdout, "\tL2_total_cache_pending_hits_from[%d] = %u\n", i, total_l2_css.pending_hits_to[i]);
            fprintf( stdout, "\tL2_total_cache_reservation_fails_from[%d] = %u\n", i, total_l2_css.res_fails_to[i]);
        }
*/
	fprintf(stdout,"total_access_local = %lld, total_miss_local = %lld, total_access_rate_local = %.4lf, total_miss_rate_local = %.4lf\n",total_access[0], total_miss[0], (total_l2_css.accesses==0)?0:(double)total_access[0]/(double)total_l2_css.accesses, (total_access[0]==0)?0:(double)total_miss[0]/(double)total_access[0]);
	fprintf(stdout,"total_access_near = %lld, total_miss_near = %lld, total_access_rate_near = %.4lf, total_miss_rate_near = %.4lf\n",total_access[1], total_miss[1], (total_l2_css.accesses==0)?0:(double)total_access[1]/(double)total_l2_css.accesses, (total_access[1]==0)?0:(double)total_miss[1]/(double)total_access[1]);
	fprintf(stdout,"total_access_remote = %lld, total_miss_remote = %lld, total_access_rate_remote = %.4lf, total_miss_rate_remote = %.4lf\n",total_access[2], total_miss[2], (total_l2_css.accesses==0)?0:(double)total_access[2]/(double)total_l2_css.accesses, (total_access[2]==0)?0:(double)total_miss[2]/(double)total_access[2]);


       if (!m_memory_config->m_L2_config.disabled() && m_memory_config->m_L2_config.get_num_lines()) {
          //L2c_print_cache_stat();
          printf("L2_total_cache_accesses = %u\n", total_l2_css.accesses);
          printf("L2_total_cache_misses = %u\n", total_l2_css.misses);
          if(total_l2_css.accesses > 0)
              printf("L2_total_cache_miss_rate = %.4lf\n", (double)total_l2_css.misses/(double)total_l2_css.accesses);
          printf("L2_total_cache_pending_hits = %u\n", total_l2_css.pending_hits);
          printf("L2_total_cache_reservation_fails = %u\n", total_l2_css.res_fails);
          printf("L2_total_cache_breakdown:\n");
          l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
          total_l2_css.print_port_stats(stdout, "L2_cache");

/*          printf("App1_L2_total_cache_accesses = %u\n", total_l2_css_app1.accesses);
          printf("App1_L2_total_cache_misses = %u\n", total_l2_css_app1.misses);
          if(total_l2_css.accesses > 0)
              printf("App1_L2_total_cache_miss_rate = %.4lf\n", (double)total_l2_css_app1.misses/(double)total_l2_css_app1.accesses);

          printf("App2_L2_total_cache_accesses = %u\n", total_l2_css_app2.accesses);
          printf("App2_L2_total_cache_misses = %u\n", total_l2_css_app2.misses);
          if(total_l2_css.accesses > 0)
              printf("App2_L2_total_cache_miss_rate = %.4lf\n", (double)total_l2_css_app2.misses/(double)total_l2_css_app2.accesses);

//////////////////
          printf("HBM CACHE Read_L2_total_cache_accesses = %lld\n", KAIN_kernel1_LLC_access);
          printf("HBM CACHE Read_L2_total_cache_misses = %lld\n", KAIN_kernel1_LLC_access-KAIN_kernel1_LLC_hit);
          if(KAIN_kernel1_LLC_access > 0)
              printf("HBM CACHE Read_L2_total_cache_miss_rate = %.4lf\n", (double)(KAIN_kernel1_LLC_access-KAIN_kernel1_LLC_hit)/(double)KAIN_kernel1_LLC_access);


          printf("HBM CACHE Write_L2_total_cache_accesses = %lld\n", KAIN_kernel2_LLC_access);
          printf("HBM CACHE Write_L2_total_cache_misses = %lld\n", KAIN_kernel2_LLC_access-KAIN_kernel2_LLC_hit);
          if(KAIN_kernel2_LLC_access > 0)
              printf("HBM CACHE Write_L2_total_cache_miss_rate = %.4lf\n", (double)(KAIN_kernel2_LLC_access-KAIN_kernel2_LLC_hit)/(double)KAIN_kernel2_LLC_access);

          printf("HBM CACHE L2_total_cache_accesses = %lld\n", KAIN_kernel1_LLC_access+KAIN_kernel2_LLC_access);
          printf("HBM CACHE L2_total_cache_misses = %lld\n", KAIN_kernel1_LLC_access-KAIN_kernel1_LLC_hit+KAIN_kernel2_LLC_access-KAIN_kernel2_LLC_hit);
          if(KAIN_kernel1_LLC_access+KAIN_kernel2_LLC_access > 0)
              printf("HBM CACHE L2_total_cache_miss_rate = %.4lf\n", (double)(KAIN_kernel1_LLC_access-KAIN_kernel1_LLC_hit+KAIN_kernel2_LLC_access-KAIN_kernel2_LLC_hit)/((double)KAIN_kernel2_LLC_access+(double)KAIN_kernel1_LLC_access));
//////////////////
*/
       }
   }

   if (m_config.gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }

#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
       m_gpgpusim_wrapper->detect_print_steady_state(1,gpu_tot_sim_insn+gpu_sim_insn);
   }
#endif


   // Interconnect power stat print
   long total_simt_to_mem=0;
   long total_mem_to_simt=0;
   long temp_stm=0;
   long temp_mts = 0;
   for(unsigned i=0; i<m_config.num_cluster(); i++){
	   m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
	   total_simt_to_mem += temp_stm;
	   total_mem_to_simt += temp_mts;
   }
   printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
   printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

   time_vector_print();
   fflush(stdout);

   // this function should not be used
   // instead, explicitly clear finished kernel
   //clear_executed_kernel_info();
}


// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const
{
   return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst)
{
    unsigned active_count = inst.active_count();
    //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
    switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
        break;
    case shared_space:
        m_stats->gpgpu_n_shmem_insn += active_count;
        break;
    case const_space:
        m_stats->gpgpu_n_const_insn += active_count;
        break;
    case param_space_kernel:
    case param_space_local:
        m_stats->gpgpu_n_param_insn += active_count;
        break;
    case tex_space:
        m_stats->gpgpu_n_tex_insn += active_count;
        break;
    case global_space:
    case local_space:
        if( inst.is_store() )
            m_stats->gpgpu_n_store_insn += active_count;
        else
            m_stats->gpgpu_n_load_insn += active_count;
        break;
    default:
        abort();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA). 
 *  
 * @param kernel 
 *    object that tells us which kernel to ask for a CTA from 
 */
std::vector<dim3> kain_Cluster0_CTA_record_K1;
std::vector<dim3> kain_Cluster0_CTA_record_K2;
unsigned shader_core_ctx::issue_block2core( kernel_info_t &kernel )
{
    set_max_cta(kernel);

    // find a free CTA context 
    unsigned free_cta_hw_id=(unsigned)-1;
    for (unsigned i=0;i<kernel_max_cta_per_shader;i++ ) {
      if( m_cta_status[i]==0 ) {
         free_cta_hw_id=i;
         break;
      }
    }
    assert( free_cta_hw_id!=(unsigned)-1 );
    mk_scheduler->start_CTA(&kernel, m_sid, free_cta_hw_id);
    ctas_in_issued_order.push_back(free_cta_hw_id);

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = m_config->get_padded_cta_size(cta_size);
    unsigned start_thread = free_cta_hw_id * padded_cta_size;
    unsigned end_thread  = start_thread +  cta_size;

    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread, false);

    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    bool context_loading = false;

    if (m_thread[start_thread] != NULL) {
      assert(m_thread[start_thread]->m_cta_info->is_ready_for_finish());
      assert(m_thread[start_thread]->get_kernel() == m_kernel);
    }
    for (unsigned i = start_thread; i<end_thread; i++) {
        assert(m_threadState[i].m_active == false);
        m_threadState[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i/m_config->warp_size;
        nthreads_in_block += SimulationInitializer::ptx_sim_init_thread(kernel,&m_thread[i],m_sid,i,cta_size-(i-start_thread),m_config->n_thread_per_shader_kain(m_sid),this,free_cta_hw_id,warp_id,m_cluster->get_gpu(), context_loading);
        m_threadState[i].m_active = true;
        warps.set( warp_id );
    }
    if (!context_loading) {
      assert(m_thread[start_thread]->m_cta_info->is_ready_for_execute());
    }
    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader_kain(m_sid)); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    if (!context_loading) {
      m_barriers.allocate_barrier(free_cta_hw_id,warps);
    }

    // initialize the SIMT stacks and fetch hardware
    if (context_loading) {
      restart_warps( free_cta_hw_id, start_thread, end_thread);
    } else {
      init_warps( free_cta_hw_id, start_thread, end_thread);
    }
    m_n_active_cta++;

    shader_CTA_count_log(m_sid, 1);
    dim3 cta_dim3 = m_thread[start_thread]->get_ctaid();
    printf("GPGPU-Sim uArch: core:%3d, cta:%2u (%d,%d,%d) initialized @(%lld,%lld)\n", m_sid, free_cta_hw_id, cta_dim3.x, cta_dim3.y, cta_dim3.z, gpu_sim_cycle, gpu_tot_sim_cycle );

    if(m_tpc == 0)
        kain_Cluster0_CTA_record_K1.push_back(cta_dim3);
    if(m_tpc == m_config->num_shader()-1)
        kain_Cluster0_CTA_record_K2.push_back(cta_dim3);


    reset_overwrite_check(free_cta_hw_id);
    m_ldst_unit->reset_overwrite_check(free_cta_hw_id);
    return free_cta_hw_id;
}

///////////////////////////////////////////////////////////////////////////////////////////
/*
void dram_t::dram_log( int task ) 
{
   if (task == SAMPLELOG) {
      StatAddSample(mrqq_Dist, que_length());   
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",id);StatDisp(mrqq_Dist);
   }
}
*/
double dram_period_kain_index = 1.0;
double min4(double a, double b, double c, double d)
{
    double smallest = min3(a,b,c);
    if (d < smallest)
        return d;
    else
        return smallest;
}


//#define KAIN_chiplet_frequency (4*1000000000.0)
//#define KAIN_chiplet_frequency (6*1000000000.0)
//#define KAIN_chiplet_frequency (1000000000.0)
//#define KAIN_chiplet_frequency (4*kain_dram_frequency)
//#define KAIN_chiplet_frequency (8*kain_dram_frequency)
//Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void)
{
   double smallest = min4(core_time,icnt_time,dram_time,chiplet_time);
   int mask = 0x00;
   if ( l2_time <= smallest ) {
      smallest = l2_time;
      mask |= L2 ;
      l2_time += m_config.l2_period;
   }
   if ( icnt_time <= smallest ) {
      mask |= ICNT;
      icnt_time += m_config.icnt_period;
   }
   if ( dram_time <= smallest ) {
      mask |= DRAM;
      dram_time += (m_config.dram_period*dram_period_kain_index);
   }
   if ( core_time <= smallest ) {
      mask |= CORE;
      core_time += m_config.core_period;
   }
   if (chiplet_time <= smallest ) {
      mask |= CHIPLET;
      chiplet_time += 1.0/KAIN_chiplet_frequency;
   }

   return mask;
}

void gpgpu_sim::issue_block2core()
{
    unsigned last_issued = m_last_cluster_issue;
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
        unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
        unsigned num = m_cluster[idx]->issue_block2core();
        if( num ) {
            m_last_cluster_issue=idx;
            m_total_cta_launched += num;
        }
    }
}

unsigned long long g_single_step=0; // set this in gdb to single step the pipeline

bool KAIN_profiling_phase1 = false;
bool KAIN_profiling_phase2 = false;
bool KAIN_profiling_phase3 = false;

std::map<std::string, KAIN_IPC> KAIN_stream1;
std::vector<float> KAIN_stream1_ipc;
//std::map<std::string, unsigned> KAIN_CTA_number_stream1_record;
unsigned CTA_finished_number_stream1 = 0;
std::string KAIN_kernel1;
bool KAIN_stream1_kernel_new_launch= false;
bool KAIN_stream1_kernel_profiling_wait_result = false;

std::map<std::string, KAIN_IPC> KAIN_stream2;
std::vector<float> KAIN_stream2_ipc;
//std::map<std::string, unsigned> KAIN_CTA_number_stream2_record;
unsigned CTA_finished_number_stream2 = 0;
std::string KAIN_kernel2;
bool KAIN_stream2_kernel_new_launch = false;
bool KAIN_stream2_kernel_profiling_wait_result = false;

std::map<std::string, KAIN_IPC> KAIN_stream3;
std::vector<float> KAIN_stream3_ipc;
std::map<std::string, unsigned> KAIN_CTA_number_stream3_record;
unsigned CTA_finished_number_stream3 = 0;
std::string KAIN_kernel3;
bool KAIN_stream3_kernel_new_launch= false;
bool KAIN_stream3_kernel_profiling = false;
bool KAIN_stream3_kernel_profiling_wait_result = false;

std::map<std::string, KAIN_IPC> KAIN_stream4;
std::vector<float> KAIN_stream4_ipc;
std::map<std::string, unsigned> KAIN_CTA_number_stream4_record;
unsigned CTA_finished_number_stream4 = 0;
std::string KAIN_kernel4;
bool KAIN_stream4_kernel_new_launch= false;
bool KAIN_stream4_kernel_profiling = false;
bool KAIN_stream4_kernel_profiling_wait_result = false;


unsigned long long KAIN_stable_cycles_THREHOLD = 20000;
unsigned long long KAIN_stable_cycles = 0;

unsigned long long kain_request_number1 = 0;

bool KAIN_stream1_H_SM = true;
bool KAIN_stream2_H_SM = true;
bool KAIN_stream3_H_SM = true;
bool KAIN_stream4_H_SM = true;


bool KAIN_Re_partition = true;

int KAIN_power_gated_count;


std::vector<int> KAIN_cluster_port_receive[8];
std::vector<int> KAIN_cluster_receive;
std::vector<int> KAIN_all_port_receive;

int kain_one_flit_contention_stall = 0;
int kain_one_flit_count = 0;
bool kain_flit_use = false;
std::vector<float> KAIN_contention_total_number;
std::vector<float> KAIN_contention_failed_number;
std::vector<float> KAIN_contention_portion;



int KAIN_mem_app1 = 0;
int KAIN_mem_app2 = 0;

unsigned long long KAIN_epoch_cycle = 0;
#define KAIN_epoch 50000000


int kain_Use_Drain_Not_Context_Switch_K1= 0;
int kain_Use_Drain_Not_Context_Switch_K2= 0;


extern std::vector<new_addr_type *> kain_page_cycle[2];

//ZSQ 20201208
unsigned last_window_accesses = 0;
unsigned last_window_misses = 0;
unsigned last_window_accesses_remote = 0;
unsigned last_window_misses_remote = 0;

void gpgpu_sim::print_window_L2(unsigned long long cur_cycle) {
        fprintf( stdout, "\n L2 cache stats in time window %lld - %lld \n", cur_cycle-1000, cur_cycle);
        struct cache_sub_stats total_css_tmp;
        struct cache_sub_stats l2_css;
        total_css_tmp.clear();
        for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++){
            m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);
            bool has_access = false;
            for (int j = 0; j < 4; j++) {
                if (l2_css.accesses_to[j] != 0) //ZSQ: mem_side L2 bank accesses_to[j] is accesses from chiplet j
                    has_access = true;
            }
            if (has_access) {
                fprintf( stdout, "\tL2_cache_bank[%d]:\n", i);
                for (int j = 0; j < 4; j++) {
                    if (l2_css.accesses_to[j] != 0)
                        fprintf( stdout, "  Access_to[%d] = %d, Miss_to[%d] = %d, Miss_rate_to[%d] = %.3lf\n",
                         j, l2_css.accesses_to[j]-last_window_accesses_from_l2_to[i][j], j, l2_css.misses_to[j]-last_window_misses_from_l2_to[i][j], j, (double)(l2_css.misses_to[j]-last_window_misses_from_l2_to[i][j]) / (double)(l2_css.accesses_to[j]-last_window_accesses_from_l2_to[i][j]));
                    //update last_window_*
                    last_window_accesses += l2_css.accesses_to[j]-last_window_accesses_from_l2_to[i][j];
                    last_window_misses += l2_css.misses_to[j]-last_window_misses_from_l2_to[i][j];
                    if (i/16 != j) {
                        last_window_accesses_remote += l2_css.accesses_to[j]-last_window_accesses_from_l2_to[i][j];
                        last_window_misses_remote += l2_css.misses_to[j]-last_window_misses_from_l2_to[i][j];
                    }
                    last_window_accesses_from_l2_to[i][j] = l2_css.accesses_to[j];
                    last_window_misses_from_l2_to[i][j] = l2_css.misses_to[j];
                }
            }
            total_css_tmp += l2_css;
            if  ((i+1)%16 == 0) {
                fprintf( stdout, "  From chiplet%d to chiplets:\n", i/16);
                for (int j = 0; j < 4; j++) {
                    fprintf( stdout, "      Access_from_to[%d][%d] = %d, Miss_from_to[%d][%d] = %d, Miss_rate_to[%d][%d] = %.3lf\n",
                     i/16, j, (total_css_tmp.accesses_to[j]-last_window_accesses_from_to[i/16][j]), i/16, j, (total_css_tmp.misses_to[j]-last_window_misses_from_to[i/16][j]), i/16, j, (double)(total_css_tmp.misses_to[j]-last_window_misses_from_to[i/16][j]) / (double)(total_css_tmp.accesses_to[j]-last_window_accesses_from_to[i/16][j]));
                    //update last_window_*
                    last_window_misses_from_to[i/16][j] = total_css_tmp.misses_to[j];
                    last_window_accesses_from_to[i/16][j] = total_css_tmp.accesses_to[j];
                }
                total_css_tmp.clear();
            }
        }
            fprintf( stdout, "L2 total access = %u, L2 total miss = %u, L2_total_miss_rate = %.3lf\n", last_window_accesses, last_window_misses, (last_window_accesses == 0)?0:(double)last_window_misses/(double)last_window_accesses);
            fprintf( stdout, "L2 total remote access = %u, L2 total remote miss = %u, L2_total_remote_miss_rate = %.3lf\n", last_window_accesses_remote, last_window_misses_remote, (last_window_accesses_remote == 0)?0:(double)last_window_misses_remote/(double)last_window_accesses_remote);
                last_window_accesses = 0;
                last_window_misses = 0;
                last_window_accesses_remote = 0;
                last_window_misses_remote = 0;

        fprintf( stdout, "\n");
  }

void gpgpu_sim::print_window_data_sharing(unsigned long long cur_cycle) {
	//ZSQ data sharing record
	fprintf( stdout, "=== data sharing record in time window %lld - %lld ===\n", cur_cycle-1000, cur_cycle);
	unsigned long long access_block_num[4] = {0,0,0,0}; //block in module i accessed in this time window
	unsigned long long shared_block_num[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}; //[i][j]: block in module i shared by j modules in this time window
	unsigned long long access_block = 0; //total number of accessed block in this time window
	unsigned long long shared_block = 0; //total number of shared block in this time window
	unsigned long long shared_block_n[4] = {0,0,0,0}; //[i]:block in module i shared in this time window
	unsigned long long shared_block_intra_module[4] = {0,0,0,0};//[i]: SMs in module i sharing block number
	unsigned long long total_access_write = 0;
	unsigned long long total_shared_module_write = 0;
	unsigned long long total_shared_sm_write = 0;

/*	for (std::map<new_addr_type, module_record>::iterator it = record_window.begin(), it_end = record_window.end(); it != it_end; ++it) {
	    access_block++;
	    for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
		  if (it->second.record[i][j]>1) {
		    shared_block_intra_module[j]++; // SMs in module j share this block 
		    if (cur_cycle==5000 || cur_cycle==20000 || cur_cycle==100000 || cur_cycle==1000000) 
			fprintf( stdout, "\tblock %llu in module %d accessed %d times in this time window by SMs from module %d\n", it->first, i, it->second.record[i][j], j);
		  }
		    int shared_by = (it->second.record[i][j]>0)?1:0+(it->second.record[i][j]>0)?1:0+(it->second.record[i][j]>0)?1:0+(it->second.record[i][j]>0)?1:0;
		    if (shared_by > 0) {
	        	access_block_num[i]++;
			shared_block_num[i][shared_by-1]++; 
			if (shared_by > 1) {
			    shared_block_n[i]++;
			    shared_block++;
			}
		        break;
		    }
		}
	    }
	}	
*/
//ZSQ 210215
        for (std::map<new_addr_type, module_record>::iterator it = record_window.begin(), it_end = record_window.end(); it != it_end; ++it) {
            access_block++;
	    if(it->second.rwtag) total_access_write++; //20210403
            access_block_number++;
            bool it_done = false;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (it->second.record[i][j]>0) { //this block is in MEM i, record[i][j=0-4]
                        access_block_num[i]++;

                        //int sm_sharing = it->second.record[i][0]+it->second.record[i][1]+it->second.record[i][2]+it->second.record[i][3]; //total SM num that access this block in this window
			int sm_sharing = 0;
			for (int l = 0; l < 128; l++) {
			    if (it->second.record_sm[i][l]>0) {
				sm_sharing ++;
			    }
			}
			if (sm_sharing == 1) sm_sharing_degree[0]++;
			else if (sm_sharing == 2) sm_sharing_degree[1]++;
			else if (sm_sharing < 5) sm_sharing_degree[2]++;
			else if (sm_sharing < 9) sm_sharing_degree[3]++;
			else if (sm_sharing < 17) sm_sharing_degree[4]++;
			else if (sm_sharing < 33) sm_sharing_degree[5]++;
			else if (sm_sharing < 65) sm_sharing_degree[6]++;
			else sm_sharing_degree[7]++;

                        if(sm_sharing>1) {
			    sm_sharing_num ++;
			    if(it->second.rwtag) total_shared_sm_write++; //20210403
			}

                        int shared_by = ((it->second.record[i][0]>0)?1:0)+((it->second.record[i][1]>0)?1:0)+((it->second.record[i][2]>0)?1:0)+((it->second.record[i][3]>0)?1:0); //total module num that access this block in this window
                        shared_block_num[i][shared_by-1]++;
                        module_sharing_degree[shared_by-1]++;
                        if (shared_by > 1) { //sharing cross modules
                            shared_block_n[i]++;
                            shared_block++;
			    if(it->second.rwtag) total_shared_module_write++; //20210403
                            module_sharing_num++;
                        }
                        //fprintf( stdout, "\trecord[%d][0]=%d, record[%d][1]=%d, record[%d][2]=%d, record[%d][3]=%d, shared_by=%d, shared_block_n[%d]=%d, shared_block=%d\n",i,it->second.record[i][0],i,it->second.record[i][1],i,it->second.record[i][2],i,it->second.record[i][3],shared_by,i,shared_block_n[i],shared_block);
                        if (cur_cycle==5000 || cur_cycle==20000 || cur_cycle==100000 || cur_cycle==1000000) {
			    fprintf( stdout, "\tblock %llu in module %d accessed %d times by SMs:", it->first, i, sm_sharing);
                            for (int l = 0; l < 128; l++) {
			        if (it->second.record_sm[i][l]>1) fprintf( stdout, " %d(%d)", l, it->second.record_sm[i][l]);
			        else if (it->second.record_sm[i][l]>0) fprintf( stdout, " %d", l);
			    }
			    fprintf( stdout, "\n");
			}
                        for (int k = j; k < 4; k++) {
                            if (it->second.record[i][k]>1) {
                                shared_block_intra_module[k]++; // SMs in module j share this block
                                //if (cur_cycle==5000 || cur_cycle==20000 || cur_cycle==100000 || cur_cycle==1000000)
                        	    //fprintf( stdout, "\tblock %llu in module %d accessed %d times in this time window by SMs from module %d\n", it->first, i, it->second.record[i][j], j);
			    }
                        }

                        it_done = true;
                        break;
                    }
                }
                if (it_done) break;
            }
        }

	fprintf( stdout, "SMs in the same module sharing: %llu %.4lf blocks for module 0, %llu %.4lf blocks for module 1, %llu %.4lf blocks for module 2, %llu %.4lf blocks for module 3.\n", shared_block_intra_module[0], (double)shared_block_intra_module[0]/(double)access_block, shared_block_intra_module[1], (double)shared_block_intra_module[1]/(double)access_block, shared_block_intra_module[2], (double)shared_block_intra_module[2]/(double)access_block, shared_block_intra_module[3], (double)shared_block_intra_module[3]/(double)access_block);
	for (int i = 0; i < 4; i++) {
	    if (access_block_num[i] == 0) fprintf( stdout, "Module MEM %d: access 0 block\n", i);
	    else fprintf( stdout, "Module MEM %d: access %llu blocks, shared %llu blocks (%.4lf): %lld shared by 2 (%.4lf), %lld shared by 3 (%.4lf), %lld shared by 4 (%.4lf)\n", i, access_block_num[i], shared_block_n[i], (double)shared_block_n[i]/(double)access_block_num[i], shared_block_num[i][1], (double)shared_block_num[i][1]/(double)access_block_num[i], shared_block_num[i][2], (double)shared_block_num[i][2]/(double)access_block_num[i], shared_block_num[i][3], (double)shared_block_num[i][3]/(double)access_block_num[i]);
	}
	if (access_block == 0) fprintf( stdout, "Total: access 0 block\n");
	else fprintf( stdout, "Total: access %llu blocks, shared %llu blocks (%.4lf)\n", access_block, shared_block, (double)shared_block/(double)access_block);

	//ZSQ 210215
	fprintf( stdout, "SM sharing degree: ");
	for (int i = 0; i < 8; i++) fprintf( stdout, "%llu, %.2lf; ", sm_sharing_degree[i], (double)sm_sharing_degree[i]/(double)access_block_number);
	fprintf( stdout, "\n");
	fprintf( stdout, "Module sharing degree: ");
        for (int i = 0; i < 4; i++) fprintf( stdout, "%llu, %.2lf; ", module_sharing_degree[i], (double)module_sharing_degree[i]/(double)access_block_number);
        fprintf( stdout, "\n");
	fprintf( stdout, "Sharing rate: %.4lf blocks shared by SMs, %.4lf blocks shared by modules\n", (double)sm_sharing_num/(double)access_block_number, (double)module_sharing_num/(double)access_block_number);
	//20210403
	fprintf( stdout, "Write rate: %.4lf blocks shared by SMs is wirte, %.4lf blocks shared by modules is write, %.4lf blocks accessed is write\n", (double)total_shared_sm_write/(double)sm_sharing_num, (double)total_shared_module_write/(double)module_sharing_num, (double)total_access_write/(double)access_block_number);

	fprintf( stdout, "\n");
	record_window.clear();
}

void gpgpu_sim::cycle() {
    int clock_mask = next_clock_domain();
    if (clock_mask & CORE) {
        //printf("KAIN page size %d\n", kain_page_cycle.size());
        int kain_mark = 0;
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < kain_page_cycle[j].size(); i++) {
                new_addr_type *tmp = kain_page_cycle[j][i];
                if ((*tmp) > 0)
                    *tmp = (*tmp) - 1;
                else
                    kain_mark = i;
            }
        //kain_page_cycle.erase(kain_page_cycle.begin());
#if REMOTE_CACHE == 1
        //ZSQ L1.5 reply out
        for (unsigned i=0;i<4;i++) {
            for (int j = 0; j < RC_BUS_WIDTH; j++) {
                 if ( !KAIN_NoC_r.remote_cache_reply_empty(i)) {
                        mem_fetch* mf = KAIN_NoC_r.remote_cache_reply_top(i);
                        if (!m_cluster[mf->get_sid()]->response_fifo_full()) {
                            mem_fetch* mf = KAIN_NoC_r.remote_cache_reply_pop(i);
                    //printf("ZSQ: remote_cache_reply_pop,");
                    //mf->print(stdout,0);
                            if(mf != NULL) {
                            m_cluster[mf->get_sid()]->response_fifo_push_back(mf);
                            }
                        }
                }
            }
        }
#endif
        // shader core loading (pop from ICNT into core) follows CORE clock
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            m_cluster[i]->icnt_cycle();
        }
#if REMOTE_CACHE == 1
        KAIN_NoC_r.remote_cache_cycle(); //ZSQ L1.5 still need to get req from miss_queue in l1_cache::cycle and push rep to response_fifo in icnt_cycle

        //ZSQ L1.5 request in
        for (unsigned i=0;i<4;i++) {
            for (int j = 0; j < RC_BUS_WIDTH; j++) {
                if ( !KAIN_NoC_r.remote_cache_request_empty(i)) {
                        mem_fetch* mf = KAIN_NoC_r.remote_cache_request_top(i);
                      if (mf!=NULL) {
                        m_cluster[mf->get_sid()]->icnt_inject_request_packet(mf);
                        KAIN_NoC_r.remote_cache_request_pop(i);
                        //printf("ZSQ: remote_cache_request_pop,");
                        //mf->print(stdout,0);
                    }
                }
            }
        }
#endif
    }

    if (clock_mask & ICNT) {
        // pop from memory controller to interconnect
#if SM_SIDE_LLC == 1
        //	printf("ZSQ: enter SM_SIDE_LLC == 1 A\n");
                for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
                    mem_fetch* mf = m_memory_sub_partition[i]->top();
                    if (mf) {
                        unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();

                        if(mf->kain_type == CONTEXT_READ_REQUEST)
                            response_size = 128;
#if BEN_OUT == 1
                        mf->set_src(m_shader_config->mem2device(i));    // soure
                        mf->set_dst(mf->get_tpc());                     // Destination
                        mf->set_next_hop(mf->get_tpc());
#endif
                        if ( ::icnt_has_buffer( m_shader_config->mem2device(i), (response_size/32+(response_size%32)?1:0)*ICNT_FREQ_CTRL*32 ) ) {
                            if (!mf->get_is_write())
                               mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
                            mf->set_status(IN_ICNT_TO_SHADER,gpu_sim_cycle+gpu_tot_sim_cycle);
                            ::icnt_push( m_shader_config->mem2device(i), mf->get_tpc(), (void*)mf, (response_size/32+(response_size%32)?1:0)*ICNT_FREQ_CTRL*32 );
                            m_memory_sub_partition[i]->pop();
                        } else {
                            gpu_stall_icnt2sh++;
        //					if(gpu_stall_icnt2sh%10000 == 0)
        //			 			printf("memory partition cannot inject packets into reply network, per 10000 times\n");
                        }
                    } else {
                       m_memory_sub_partition[i]->pop();
                    }
                }
        //	printf("ZSQ: leave SM_SIDE_LLC == 1 A\n");
#endif

#if SM_SIDE_LLC == 0
#if BEN_OUTPUT == 1
        std::ostringstream out;
#endif
        for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
            mem_fetch *mf = m_memory_sub_partition[i]->top();
            if (mf) {
                unsigned response_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
                if (mf->kain_type == CONTEXT_READ_REQUEST)
                    response_size = 128;
                if (mf->get_sid() / 32 != mf->get_chip_id() / 8) { //remote, inter_icnt
                    unsigned to_module = 192 + mf->get_sid() / 32;

#if BEN_OUTPUT == 1
                    mf->set_dst(to_module);
                    mf->set_src(192 + mf->get_chip_id() / 8);
                    mf->set_chiplet(mf->get_chip_id() / 8);
                    mf->set_next_hop(to_module);
#endif
                    if (INTER_TOPO == 1 && (mf->get_sid() / 32 + mf->get_chip_id() / 8) % 2 == 0) //ring, forward
                        to_module = 192 + (mf->get_sid() / 32 + 1) % 4;
                    //ZSQ0126
                    if (::icnt_has_buffer(192 + mf->get_chip_id() / 8, response_size)) {
                        if (!mf->get_is_write())
                            mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                        mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
                        ::icnt_push(192 + mf->get_chip_id() / 8, to_module, (void *) mf, response_size);
                        m_memory_sub_partition[i]->pop();
#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle >= 1000000) {
                            out << "L2_icnt_pop\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize: "
                                << response_size << "\n";
                            rep3->apply(out.str().c_str());
                        }
#endif
                    }
                    else {
                        gpu_stall_icnt2sh++;
                    }
                }
                else { //local
                    if (::icnt_has_buffer(m_shader_config->mem2device(i),
                                          (response_size / 32 + (response_size % 32) ? 1 : 0) * ICNT_FREQ_CTRL * 32)) {
                        if (!mf->get_is_write())
                            mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                        mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
#if BEN_OUTPUT == 1
                        mf->set_src(m_shader_config->mem2device(i));
                        mf->set_dst(mf->get_tpc());
                        mf->set_next_hop(mf->get_tpc());
                        mf->set_chiplet(m_shader_config->mem2device(i));
#endif
                        ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), (void *) mf,
                                    (response_size / 32 + (response_size % 32) ? 1 : 0) * ICNT_FREQ_CTRL * 32);
/*#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle >= 1000000) {
                            out << "L2_icnt_pop\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize: "
                                << response_size << "\tlocal reply\n";
                            rep3->apply(out.str().c_str());
                        }
#endif*/
                        m_memory_sub_partition[i]->pop();
                    }
                    else {
                        gpu_stall_icnt2sh++;
                    }
                }
            }
            else {
                m_memory_sub_partition[i]->pop();
            }
        }
#endif
    }

    if (clock_mask & DRAM) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {

            m_memory_partition_unit[i]->dram_cycle(); // Issue the dram command (scheduler + delay model)
            // Update performance counters for DRAM
            /*
            m_memory_partition_unit[i]->set_dram_power_stats(m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
                           m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
                           m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
            */
        }
    }

    if (clock_mask & L2) {
        m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
#if BEN_OUTPUT == 1
        std::ostringstream out;
#endif
        for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
            //move memory request from interconnect into memory partition (if not backed up)
            //Note:This needs to be called in DRAM clock domain if there is no L2 cache in the system
            if (m_memory_sub_partition[i]->full()) {
                gpu_stall_dramfull++;
//			 if(gpu_stall_dramfull%10000 == 0)
//			 	printf("memory partition is full, so cannot accept packets from request network, per 10000 times\n");
            }
            else {
#if SM_SIDE_LLC == 0
                if (KAIN_NoC_r.get_inter_icnt_pop_llc_turn(i)) { //pop from inter_icnt_pop_llc
                    if (!KAIN_NoC_r.inter_icnt_pop_llc_empty(i)) {
                        mem_fetch *mf;
                        inter_delay_t *x6 = KAIN_NoC_r.inter_icnt_pop_llc_pop(i);
                        if (x6) {
                            mf = x6->req;
                            mf->set_icnt_cycle(x6->ready_cycle);
                            mf->set_chiplet(i / 16);
                            if (mf != NULL) {
                                unsigned request_size;
                                if(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_ACK)
                                    request_size = mf->get_ctrl_size();
                                else if(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_REQUEST)
                                    request_size = mf->size();
                                //m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle + 32);
                                m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
                                KAIN_NoC_r.set_inter_icnt_pop_llc_turn(i);
#if BEN_OUTPUT == 1
                                if(gpu_sim_cycle >= 1000000) {
                                    out << "rop push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                        "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                        << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                                        << "\tsize: "
                                        << request_size << "\n";
                                    rep3->apply(out.str().c_str());
                                }
#endif
                            }
                        }
                    }
                    else {
                        mem_fetch *mf = (mem_fetch *) icnt_pop(m_shader_config->mem2device(i));
                        //if(mf != NULL && mf->kain_type == CONTEXT_WRITE_REQUEST)
                            //printf("KAIN KAIN received the write reuquest %lld, mf id %d\n",kain_request_number1++,mf->get_request_uid());
                        if (mf != NULL) {
                            m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
                            unsigned request_size;
                            if(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_ACK)
                                request_size = 8;
                            else if(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_REQUEST)
                                request_size = 136;
/*#if BEN_OUTPUT == 1
                            if(gpu_sim_cycle >= 1000000) {
                                out << "rop push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                    "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                    << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                                    << "\tsize: "
                                    << request_size << "\tLocal packet\n";
                                rep3->apply(out.str().c_str());
                            }
#endif*/
                        }
                    }
                }
                else {
                    mem_fetch *mf = (mem_fetch *) icnt_pop(m_shader_config->mem2device(i));
                    if (mf == NULL && !KAIN_NoC_r.inter_icnt_pop_llc_empty(i)) {
                        inter_delay_t *x7 = KAIN_NoC_r.inter_icnt_pop_llc_pop(i);
                        if (x7) {
                            mf = x7->req;
                            mf->set_icnt_cycle(x7->ready_cycle);

                            if (mf != NULL) { //ZSQ0123
                                m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle); //ZSQ0125
                                unsigned request_size;
                                if(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_ACK)
                                    request_size = 8;
                                else if(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_REQUEST)
                                    request_size = 136;
#if BEN_OUTPUT == 1
                                if(gpu_sim_cycle >= 1000000) {
                                    out << "rop push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                        "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                        << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet()
                                        << "\tsize: "
                                        << request_size << "\n";
                                    rep3->apply(out.str().c_str());
                                }
#endif
                            }
                        }
                    }
                    else if (mf != NULL) {
                        //m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle + 32);
                        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
                        KAIN_NoC_r.set_inter_icnt_pop_llc_turn(i);
                        unsigned request_size;
                        if(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_ACK)
                            request_size = 8;
                        else if(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_REQUEST)
                            request_size = 136;
#if BEN_OUTPUT == 1
                        /*if(gpu_tot_sim_cycle >= 1000000) {
                            out << "rop push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize: "
                                <<
                                request_size << "\tLocal packet\n";
                            rep3->apply(out.str().c_str());
                        }*/
#endif
                    }
                }

#endif

#if SM_SIDE_LLC == 1
                //		  printf("ZSQ: enter SM_SIDE_LLC == 1 B\n");
                                  mem_fetch* mf = (mem_fetch*) icnt_pop( m_shader_config->mem2device(i) );
                                  if (mf != NULL){ //ZSQ0123
                                        m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
                                        unsigned request_size;
                                        if(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_ACK)
                                            request_size = 8;
                                        else if(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_REQUEST)
                                            request_size = 136;
#if BEN_OUTPUT == 1
                                        mf->set_chiplet(i/16);
                                        if(gpu_sim_cycle >= 1000000){
                                            out << "rop push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                                "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                                << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() <<
                                                "\tsize: " << request_size << "\n";
                                            rep3->apply(out.str().c_str());
                                        }
#endif
                                  }
#endif
            }
//ZSQ 210223
#if REMOTE_CACHE == 1
            if ((gpu_sim_cycle+gpu_tot_sim_cycle)%2) {
              m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle+gpu_tot_sim_cycle);
                  m_memory_sub_partition[i]->accumulate_L2cache_stats(m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
            }
#endif
#if REMOTE_CACHE == 0
            m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
            m_memory_sub_partition[i]->accumulate_L2cache_stats(
                    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
#endif
        }
        scheduler->l2_cache_cycle();
    }

    if (clock_mask & ICNT) {
        icnt_transfer();
    }

    if (clock_mask & CORE) {
        // L1 cache + shader core pipeline stages
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
                m_cluster[i]->core_cycle();
                *active_sms += m_cluster[i]->get_n_active_sms();
            }
            // Update core icnt/cache stats for GPUWattch
            m_cluster[i]->get_icnt_stats(m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
                                         m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
            m_cluster[i]->get_cache_stats(m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
        }
        float temp = 0;
        for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
            temp += m_shader_stats->m_pipeline_duty_cycle[i];
        }
        temp = temp / m_shader_config->num_shader();
        *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
        //cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;

        if (g_single_step && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
            asm("int $03");
        }

//ZSQ0126 forward ready requests
#if INTER_TOPO == 1
#if BEN_OUTPUT == 1
        std::ostringstream out3;
#endif
        for (int i = 0; i < 4; i++) {
            while (!KAIN_NoC_r.forward_waiting_empty(i)) { //has ready request/reply
                inter_delay_t *x = KAIN_NoC_r.forward_waiting_pop(i);
                if(x) {
                    mem_fetch *tmp = x->req;
                    tmp->set_icnt_cycle(x->ready_cycle);
                    unsigned tmp_size;
                    if (tmp->get_type() == READ_REPLY || tmp->get_type() == WRITE_ACK) {//reply
#if BEN_OUTPUT == 1
                        tmp->set_dst(192 + tmp->get_sid() / 32);
                        tmp->set_src(192 + i);
                        tmp->set_chiplet(i);
                        tmp->set_next_hop(192 + tmp->get_sid() / 32);
#endif
                        if (!tmp->get_is_write() && !tmp->isatomic())
                            tmp_size = tmp->size();
                        else
                            tmp_size = tmp->get_ctrl_size();
                        ::icnt_push(192 + i, 192 + tmp->get_sid() / 32, tmp, tmp_size);
#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle >= 1000000) {
                            out3 << "FW pop\tsrc: " << tmp->get_src() << "\tdst: " << tmp->get_dst() <<
                                 "\tID: " << tmp->get_request_uid() << "\ttype: " << tmp->get_type()
                                 << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << tmp->get_chiplet() << "\tsize: "
                                 << tmp_size << "\n";
                            rep3->apply(out3.str().c_str());
                        }
#endif
                    } else { //request
                        if (!tmp->get_is_write() && !tmp->isatomic())
                            tmp_size = tmp->get_ctrl_size();
                        else
                            tmp_size = tmp->size();
#if BEN_OUTPUT == 1
                        tmp->set_dst(192 + tmp->get_chip_id() / 8);
                        tmp->set_src(192 + i);
                        tmp->set_next_hop(192 + tmp->get_chip_id() / 8);

#endif
                        ::icnt_push(192 + i, 192 + tmp->get_chip_id() / 8, tmp, tmp_size);
#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle >= 1000000) {
                            out3 << "FW pop\tsrc: " << tmp->get_src() << "\tdst: " << tmp->get_dst() <<
                                 "\tID: " << tmp->get_request_uid() << "\ttype: " << tmp->get_type()
                                 << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << tmp->get_chiplet() << "\tsize: "
                                 << tmp_size << "\n";
                            rep3->apply(out3.str().c_str());
                        }
#endif
                    }
                }
            }
        }
#endif
//ZSQ0126

        gpu_sim_cycle++;
        unsigned long long cur_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
        if ((cur_cycle >= 100000 && cur_cycle <= 10000000 && cur_cycle % 100000 == 0)) {
            printf("ZSQ print stat: gpu_tot_sim_cycle = %lld\n", cur_cycle);
            gpu_print_stat();
            fflush(stdout);
        }
        //ZSQ 20201208
        if (((cur_cycle <= 100000) && (cur_cycle % 1000 == 0)) ||
            ((cur_cycle >= 1000000) && (cur_cycle <= 1100000) && (cur_cycle % 1000 == 0))) {
            //print_window_L2(cur_cycle);

        }
        if(cur_cycle % 10000 == 0){
            ::icnt_display_stats();
        }

        if (cur_cycle == 20000)
            printf("ZSQ RWrate: gpu_tot_sim_cycle = %lld, rate = %.4lf\n", cur_cycle, (double) llc_w / (double) (llc_w + llc_r));
        for (std::set<kernel_info_t *>::iterator it = m_executing_kernels.begin(), it_end = m_executing_kernels.end();
             it != it_end; ++it) {
            (*it)->get_parent_process()->inc_cycles();
        }

        std::vector<unsigned> scheduled_num_ctas;
        scheduled_num_ctas.reserve(m_shader_config->num_shader());
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
            for (unsigned j = 0; j < m_shader_config->n_simt_cores_per_cluster; ++j) {
                scheduled_num_ctas.push_back(m_cluster[i]->get_core(j)->get_n_active_cta());
            }
        }
        scheduler->core_cycle(scheduled_num_ctas);

        bool KAIN_in_switch_context = false;
        for (int i = 0; i < 80; i++) {
            KAIN_in_switch_context = KAIN_in_switch_context || m_cluster[i]->KAIN_is_preempting();
        }
        if (KAIN_in_switch_context == false) {
            KAIN_epoch_cycle++;
        }

        if (KAIN_epoch_cycle == 1) {
            printf("Context Switch Over, CYCLE %lld\n", gpu_sim_cycle + gpu_tot_sim_cycle);

            kain_Cluster0_CTA_record_K1.clear();
            kain_Cluster0_CTA_record_K2.clear();
            kain_Use_Drain_Not_Context_Switch_K1 = 0;
            kain_Use_Drain_Not_Context_Switch_K2 = 0;

            //clear bw_utilization
            kain_cycles_HBM_app1 = 0;
            kain_cycles_HBM_app2 = 0;
            kain_write_back_cycles = 0;
            kain_cycles_HBM_total = 0;


            //clear row-buffer locality
            kain_row_hit_app1 = 0;
            kain_row_hit_app2 = 0;
            kain_row_miss_app1 = 0;
            kain_row_miss_app2 = 0;

            //clear CACHE
            for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++)
                m_memory_sub_partition[i]->clear_L2cache_sub_stats_kain();


            KAIN_kernel1_LLC_access = 0;
            KAIN_kernel1_LLC_hit = 0;
            KAIN_kernel2_LLC_access = 0;
            KAIN_kernel2_LLC_hit = 0;

            //clear warp
            extern long long kain_warp_inst_app1;
            extern long long kain_warp_inst_app2;
            kain_warp_inst_app1 = 0;
            kain_warp_inst_app2 = 0;
        }

        if (KAIN_epoch_cycle == KAIN_epoch) {
            KAIN_epoch_cycle = 0;
            //Real bw_app1, bw_app2
            float bw_app1 = 0.0;
            float bw_app2 = 0.0;
            //  for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
            printf("App1_write_hit %lld, App2_write_hit %lld, kain_write_back_cycles %lld, kain_cycles_HBM_total %lld\n",
                   App1_write_hit, App2_write_hit, kain_write_back_cycles, kain_cycles_HBM_total);
            printf("App1 kain cycles %lld, App2 kain cycles %lld\n", kain_cycles_HBM_app1, kain_cycles_HBM_app2);

            if ((App1_write_hit + App2_write_hit) != 0) {
                bw_app1 += (float) kain_cycles_HBM_app1 / (float) kain_cycles_HBM_total +
                           (float) (App1_write_hit) / (float) (App1_write_hit + App2_write_hit) *
                           (float) (kain_write_back_cycles) / (float) kain_cycles_HBM_total;
                bw_app2 += (float) kain_cycles_HBM_app2 / (float) kain_cycles_HBM_total +
                           (float) (App2_write_hit) / (float) (App1_write_hit + App2_write_hit) *
                           (float) (kain_write_back_cycles) / (float) kain_cycles_HBM_total;
            } else {
                bw_app1 += (float) kain_cycles_HBM_app1 / (float) kain_cycles_HBM_total;
                bw_app2 += (float) kain_cycles_HBM_app2 / (float) kain_cycles_HBM_total;
            }
            //  bw_app1 = bw_app1 / (float)m_memory_config->m_n_mem;
            //  bw_app2 = bw_app2 / (float)m_memory_config->m_n_mem;


            //row locality app1, app2 and predicate its isolate bw_utilization

            float locality_app1 = (float) kain_row_hit_app1 / (float) (kain_row_hit_app1 + kain_row_miss_app1);
            float locality_app2 = (float) kain_row_hit_app2 / (float) (kain_row_hit_app2 + kain_row_miss_app2);

            float bw_app1_predicate = (((locality_app1) * 0.538719553335059 + 0.216906174332426));
            float bw_app2_predicate = (((locality_app2) * 0.538719553335059 + 0.216906174332426));

            //Check memory-intensive or compute-intensive of an app

            printf("locality_app1 %lf, locality_app2 %lf\n", locality_app1, locality_app2);
            printf("bw_app1 %lf, bw_app2 %lf\n", bw_app1, bw_app2);
            printf("bw_app1_predicate %lf, bw_app2_redicate %lf\n", bw_app1_predicate, bw_app2_predicate);
            fflush(stdout);


            struct cache_sub_stats l2_css_app1;
            struct cache_sub_stats total_l2_css_app1;
            l2_css_app1.clear();
            total_l2_css_app1.clear();

            struct cache_sub_stats l2_css_app2;
            struct cache_sub_stats total_l2_css_app2;
            l2_css_app2.clear();
            total_l2_css_app2.clear();

            for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
                for (unsigned j = 0; j < 80; j++) {
                    if (Stream1_SM[j] == true) {
                        m_memory_sub_partition[i]->get_L2cache_sub_stats_kain(j, l2_css_app1);
                        total_l2_css_app1 += l2_css_app1;
                    }
                }
                for (unsigned j = 0; j < 80; j++) {
                    if (Stream2_SM[j] == true) {
                        m_memory_sub_partition[i]->get_L2cache_sub_stats_kain(j, l2_css_app2);
                        total_l2_css_app2 += l2_css_app2;
                    }
                }

                m_memory_sub_partition[i]->clear_L2cache_sub_stats_kain();
            }

            unsigned kain_app1_miss = total_l2_css_app1.misses;
            unsigned kain_app2_miss = total_l2_css_app2.misses;

            long long kain_app1_miss_mimic = KAIN_kernel1_LLC_access - KAIN_kernel1_LLC_hit;
            long long kain_app2_miss_mimic = KAIN_kernel2_LLC_access - KAIN_kernel2_LLC_hit;
            KAIN_kernel1_LLC_access = 0;
            KAIN_kernel1_LLC_hit = 0;
            KAIN_kernel2_LLC_access = 0;
            KAIN_kernel2_LLC_hit = 0;


            extern long long kain_warp_inst_app1;
            extern long long kain_warp_inst_app2;
            printf("App1, miss %u, miss_mimic %lld, warp inst %lld\n", kain_app1_miss, kain_app1_miss_mimic,
                   kain_warp_inst_app1);
            printf("App2, miss %u, miss_mimic %lld, warp inst %lld\n", kain_app2_miss, kain_app2_miss_mimic,
                   kain_warp_inst_app2);

            float slowdown_app1;
            float slowdown_app2;

            if ((float) kain_app1_miss_mimic / (float) kain_warp_inst_app1 * 2 * 80 * 1000000000 * 128 >
                900000000000 * bw_app1_predicate)//Memory
            {
                slowdown_app1 = ((float) bw_app1_predicate / kain_app1_miss_mimic) / (bw_app1 / kain_app1_miss);
            } else {
                unsigned total_SM = 80;
                unsigned total_SM_app1 = 0;
                for (unsigned j = 0; j < 80; j++) {
                    if (Stream1_SM[j] == true)
                        total_SM_app1++;
                }
                slowdown_app1 = (float) total_SM / total_SM_app1;
            }

            if ((float) kain_app2_miss_mimic / (float) kain_warp_inst_app2 * 2 * 80 * 1000000000 * 128 >
                900000000000 * bw_app2_predicate)//Memory
            {
                slowdown_app2 = ((float) bw_app2_predicate / kain_app2_miss_mimic) / (bw_app2 / kain_app2_miss);
            } else {
                unsigned total_SM = 80;
                unsigned total_SM_app2 = 0;
                for (unsigned j = 0; j < 80; j++) {
                    if (Stream2_SM[j] == true)
                        total_SM_app2++;
                }
                slowdown_app2 = (float) total_SM / total_SM_app2;
            }
            kain_warp_inst_app1 = 0;
            kain_warp_inst_app2 = 0;

            printf("Slowdown app1 %lf, Slowdown app2 %lf\n", slowdown_app1, slowdown_app2);

            float fairness_kain;
            if (slowdown_app1 < slowdown_app2)
                fairness_kain = slowdown_app1 / slowdown_app2;
            else
                fairness_kain = slowdown_app2 / slowdown_app1;

            printf("Current Fairness %lf\n", fairness_kain);
            if (fairness_kain < 0.9) {
                //DO the SM Repartition

                unsigned total_SM_app1 = 0;
                unsigned total_SM_app2 = 0;
                for (unsigned j = 0; j < 80; j++) {
                    if (Stream1_SM[j] == true)
                        total_SM_app1++;
                    else if (Stream2_SM[j] == true)
                        total_SM_app2++;
                }
                assert(total_SM_app1 + total_SM_app2 == 80);
                float STP_app1 = 1 / slowdown_app1;
                float STP_app2 = 1 / slowdown_app2;


                float small_STP;
                float big_STP;
                float small_SM_count;
                float big_SM_count;
                float gradient_small = STP_app1 / (float) total_SM_app1;
                float gradient_big = STP_app2 / (float) total_SM_app1;

                if (STP_app1 < STP_app2) {
                    small_STP = STP_app1;
                    big_STP = STP_app2;
                    small_SM_count = (float) total_SM_app1;
                    big_SM_count = (float) total_SM_app2;
                } else {
                    small_STP = STP_app2;
                    big_STP = STP_app1;
                    small_SM_count = (float) total_SM_app2;
                    big_SM_count = (float) total_SM_app1;
                }

                gradient_small = (1.0 - small_STP) / (80.0 - small_SM_count);
                gradient_big = (big_STP) / (big_SM_count);
                std::vector<int> Drain_list;
                Drain_list.clear();


                int SM_count_change = 0;
                for (; SM_count_change < big_SM_count; SM_count_change++) {
                    if (small_STP + gradient_small * SM_count_change > big_STP - gradient_big * SM_count_change)
                        break;
                }

                if (small_STP == STP_app1) {
                    printf("Previsou SM count, App1 %d, App2 %d\n", total_SM_app1, total_SM_app2);
                    printf("Next SM count, App1 %d, App2 %d\n", total_SM_app1 + SM_count_change,
                           total_SM_app2 - SM_count_change);

                    for (int j = total_SM_app1; j < total_SM_app1 + SM_count_change; j++) {
                        assert(Stream2_SM[j] == true);
                        Stream2_SM[j] = false;
                        Stream1_SM[j] = true;
                        Drain_list.push_back(j);
                    }
                } else {
                    printf("Previous SM count, App1 %d, App2 %d\n", total_SM_app1, total_SM_app2);
                    printf("Next SM count, App1 %d, App2 %d\n", total_SM_app1 - SM_count_change,
                           total_SM_app2 + SM_count_change);

                    for (int j = total_SM_app1 - 1; j >= total_SM_app1 - SM_count_change; j--) {
                        assert(Stream1_SM[j] == true);
                        Stream1_SM[j] = false;
                        Stream2_SM[j] = true;
                        Drain_list.push_back(j);
                    }
                }

                for (int i = 0; i < Drain_list.size(); i++) {
                    const unsigned sid = Drain_list[i];
                    unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
                    unsigned core_id = m_shader_config->sid_to_cid(sid);
                    shader_core_ctx *shader_core = m_cluster[cluster_id]->get_core(core_id);

                    unsigned kain_stream_number;
                    if (shader_core->get_kernel() != NULL) {
                        if (shader_core->get_kernel()->get_kain_stream_id() == 1) {
                            printf("K1, Cuurent CTA %d, max cta per shader %d\n", kain_Use_Drain_Not_Context_Switch_K1,
                                   shader_core->get_kernel()->get_max_cta_per_shader(0));
                            printf("K1, shader outside atomic insts %d\n", shader_core->KAIN_atomic_count());
                            if (kain_Use_Drain_Not_Context_Switch_K1 >
                                shader_core->get_kernel()->get_max_cta_per_shader(0) ||
                                shader_core->KAIN_atomic_count()) {
                                printf("K1 cluster ID %d Draining, finished CTA numbe\n", cluster_id);
                                assert(!shader_core->is_preempting());
                                for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
                                    if (shader_core->get_active_threads_for_cta(i) > 0) {
                                        shader_core->drain_cta(i);
                                    }
                                }
                            } else {
                                printf("K1 cluster ID %d Context switch, Cycle %lld\n", cluster_id,
                                       gpu_sim_cycle + gpu_tot_sim_cycle);
                                shader_core->switch_core();
                            }
                        } else if (shader_core->get_kernel()->get_kain_stream_id() == 2) {
                            printf("K2, Cuurent CTA %d, max cta per shader %d\n", kain_Use_Drain_Not_Context_Switch_K2,
                                   shader_core->get_kernel()->get_max_cta_per_shader(m_config.num_shader() - 1));
                            printf("K2, shader outside atomic insts %d\n", shader_core->KAIN_atomic_count());
                            if (kain_Use_Drain_Not_Context_Switch_K2 >
                                shader_core->get_kernel()->get_max_cta_per_shader(m_config.num_shader() - 1) ||
                                shader_core->KAIN_atomic_count()) {
                                printf("K2 cluster ID %d Draining, finished CTA numbe\n", cluster_id);
                                assert(!shader_core->is_preempting());
                                for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
                                    if (shader_core->get_active_threads_for_cta(i) > 0) {
                                        shader_core->drain_cta(i);
                                    }
                                }
                            } else {
                                printf("K2 cluster ID %d Context switch, Cycle %lld\n", cluster_id,
                                       gpu_sim_cycle + gpu_tot_sim_cycle);
                                shader_core->switch_core();
                            }
                        } else
                            assert(0);
                    }
                }
            }
        }
/*
        for(int i = 0; i < Drain_list.size(); i++) 
        {    
                const unsigned sid = Drain_list[i];
                unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
                unsigned core_id = m_shader_config->sid_to_cid(sid);
                shader_core_ctx* shader_core = m_cluster[cluster_id]->get_core(core_id);
     
                unsigned kain_stream_number;
                if( shader_core->get_kernel()!= NULL)
                {
                    if(shader_core->get_kernel()->get_kain_stream_id() == 1)
                        kain_stream_number = CTA_finished_number_stream1;
                    if(shader_core->get_kernel()->get_kain_stream_id() == 2) 
                        kain_stream_number = CTA_finished_number_stream2;
                    if(shader_core->get_kernel()->get_kain_stream_id() == 3) 
                        kain_stream_number = CTA_finished_number_stream3;
                    if(shader_core->get_kernel()->get_kain_stream_id() == 4) 
                        kain_stream_number = CTA_finished_number_stream4;
     
                    if(kain_stream_number == 0)
                    //if(1)
                    {
                        printf("cluster ID %d Context switch\n",cluster_id);
                        shader_core->switch_core();
                    }
                    else
                    {
                        printf("cluster ID %d Draining, finished CTA number %d\n",cluster_id,kain_stream_number);
                        assert(!shader_core->is_preempting());
                        for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
                            if (shader_core->get_active_threads_for_cta(i) > 0) { 
                                shader_core->drain_cta(i);
                            }
                        }
                    }
                }
           }
*/
        static int init_del_cuodump_kain = 0;
        if (init_del_cuodump_kain == 0 && gpu_sim_cycle + gpu_tot_sim_cycle > 2000) {
            Power_gating_number_cycles.clear();
            init_del_cuodump_kain = 1;
            system("rm ./*cuobjdump*");
        }

/*kain comment begin

      // now let's see if we need to preempt or cancel preemption
      std::vector<kernel_info_t*> kernels_with_remove = scheduler->check_for_removal();
      for (std::vector<kernel_info_t*>::iterator it = kernels_with_remove.begin(), it_end = kernels_with_remove.end();
           it != it_end; ++it) {
        kernel_info_t* kernel = *it;
        unsigned num_remove = scheduler->get_num_remove_required(kernel);
        assert(num_remove > 0);

        std::vector<preemption_info_t> remove_shaders = scheduler->find_shaders_to_remove(kernel, num_remove);
        assert(remove_shaders.size() <= num_remove);
        num_remove = remove_shaders.size();
        for (std::vector<preemption_info_t>::const_iterator cit = remove_shaders.begin(), cit_end = remove_shaders.end();
             cit != cit_end; ++cit) {
          const unsigned sid = cit->get_sid();
          unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
          unsigned core_id = m_shader_config->sid_to_cid(sid);
          shader_core_ctx* shader_core = m_cluster[cluster_id]->get_core(core_id);
          assert(!shader_core->is_preempting());
          assert(shader_core->get_n_active_cta() == cit->get_preempting_ctas());
          for (unsigned i = 0; i < MAX_CTA_PER_SHADER; ++i) {
            if (shader_core->get_active_threads_for_cta(i) > 0) {
              if (cit->is_cta_draining(i)) {
                shader_core->drain_cta(i);
              } else if (cit->is_cta_switching(i)) {
                shader_core->context_save_cta(i);
              } else if (cit->is_cta_flushing(i)) {
                shader_core->flush_cta(i);
              } else {
                assert(false && "Removing, but not draining nor migrating!!");
              }
            }
          }
          --num_remove;
        }

        assert(num_remove == 0);
      }

      // now let's see if we need to cancel preemption
      std::vector<kernel_info_t*> kernels_with_cancel = scheduler->check_for_cancel_removal();
      for (std::vector<kernel_info_t*>::iterator it = kernels_with_cancel.begin(), it_end = kernels_with_cancel.end();
           it != it_end; ++it) {
        kernel_info_t* kernel = *it;
        const unsigned num_cancel = scheduler->get_num_cancel_remove_required(kernel);
        assert(num_cancel > 0);

        std::vector<unsigned> cancel_shaders = scheduler->find_shaders_to_cancel(kernel, num_cancel);
        assert(cancel_shaders.size() <= num_cancel);
        for (std::vector<unsigned>::const_iterator cit = cancel_shaders.begin(), cit_end = cancel_shaders.end();
             cit != cit_end; ++cit) {
          unsigned cluster_id = m_shader_config->sid_to_cluster(*cit);
          unsigned core_id = m_shader_config->sid_to_cid(*cit);
          shader_core_ctx* shader_core = m_cluster[cluster_id]->get_core(core_id);
          assert(shader_core->is_draining());
          shader_core->cancel_drain();
        }
      }
kain comment end*/

        if (g_interactive_debugger_enabled)
            gpgpu_debug();

        // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
        if (m_config.g_power_simulation_enabled) {
            mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper, m_power_stats,
                        m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                        gpu_sim_insn);
        }
#endif

        issue_block2core();

        // Depending on configuration, flush the caches once all of threads are completed.
        int all_threads_complete = 1;
        if (m_config.gpgpu_flush_l1_cache) {
            for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
                if (m_cluster[i]->get_not_completed() == 0)
                    m_cluster[i]->cache_flush();
                else
                    all_threads_complete = 0;
            }
        }

        if (m_config.gpgpu_flush_l2_cache) {
            if (!m_config.gpgpu_flush_l1_cache) {
                for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
                    if (m_cluster[i]->get_not_completed() != 0) {
                        all_threads_complete = 0;
                        break;
                    }
                }
            }

            if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
                printf("Flushed L2 caches...\n");
                if (m_memory_config->m_L2_config.get_num_lines()) {
                    int dlc = 0;
                    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
                        dlc = m_memory_sub_partition[i]->flushL2();
                        assert(dlc == 0); // need to model actual writes to DRAM here
                        printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
                    }
                }
            }
        }

        if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
            time_t days, hrs, minutes, sec;
            time_t curr_time;
            time(&curr_time);
            unsigned long long elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
            if ((elapsed_time - last_liveness_message_time) >= m_config.liveness_message_freq) {
                days = elapsed_time / (3600 * 24);
                hrs = elapsed_time / 3600 - 24 * days;
                minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
                sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));
                printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                       gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
                       (double) gpu_sim_insn / (double) gpu_sim_cycle,
                       (unsigned) ((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                       (unsigned) days, (unsigned) hrs, (unsigned) minutes, (unsigned) sec,
                       ctime(&curr_time));
                fflush(stdout);
                last_liveness_message_time = elapsed_time;
            }
            visualizer_printstat();
            m_memory_stats->memlatstat_lat_pw();
            if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0)) {
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
                    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
                        m_memory_partition_unit[i]->print_stat(stdout);
                    printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
                    printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
                }
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
                    shader_print_runtime_stat(stdout);
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
                    shader_print_l1_miss_stat(stdout);
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
                    shader_print_scheduler_stat(stdout, false);
            }
        }

        if (!(gpu_sim_cycle % 10000)) {
            // deadlock detection
            if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn && !has_context_switching_core()) {
                gpu_deadlock = true;
            } else {
                last_gpu_sim_insn = gpu_sim_insn;
            }
        }
        try_snap_shot(gpu_sim_cycle);
        spill_log_nano to_file(stdout, 0, gpu_sim_cycle);
    }

    if (clock_mask & ICNT) {
#if SM_SIDE_LLC == 1
                for (unsigned i = 0; i < 4; i++){
                mem_fetch *mf = (mem_fetch*) ::icnt_pop(192+i);
                if (mf != NULL && INTER_TOPO == 0){ //ZSQ0126, 0 for full connection
                    unsigned _mid = mf->get_chip_id();
                    unsigned _subid = mf->get_sub_partition_id();
                icnt_pop_inter++;
        /*		if (mf->get_chip_id()/8 != i && !m_memory_sub_partition[_subid]->full()){ //reply, push to LLC
                     m_memory_sub_partition[_subid]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
                } else if (mf->get_chip_id()/8 == i && m_memory_partition_unit[_mid]->dram_latency_avaliable()){ //request, push to dram_latency_queue
                    m_memory_partition_unit[_mid]->receive_inter_icnt(mf);
                }
        */
                if (mf->get_chip_id()/8 != i && !KAIN_NoC_r.inter_icnt_pop_llc_full(_subid)){ //reply, will push to LLC
                    KAIN_NoC_r.inter_icnt_pop_llc_push(mf, _subid);
                    icnt_pop_inter_llc++;
                } else if (mf->get_chip_id()/8 == i && !KAIN_NoC_r.inter_icnt_pop_mem_full(_mid)){ //request, will push to dram_latency_queue
                        KAIN_NoC_r.inter_icnt_pop_mem_push(mf, _mid);
                    icnt_pop_inter_mem++;
                }
                } else if (mf != NULL && INTER_TOPO == 1) { //ZSQ0126, 1 for ring, forwarding if not neighbor
                    unsigned _mid = mf->get_chip_id();
                    unsigned _subid = mf->get_sub_partition_id();
                    if (mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK) { //reply
                        if (i == mf->get_sid()/32 && !KAIN_NoC_r.inter_icnt_pop_llc_full(_subid)) //arrive
                        KAIN_NoC_r.inter_icnt_pop_llc_push(mf, _subid);
                        else if (i != mf->get_sid()/32 && !KAIN_NoC_r.forward_waiting_full(i))//forward
                        KAIN_NoC_r.forward_waiting_push(mf, i);
                    }
                    else { //request
                        if (i == mf->get_chip_id()/8 && !KAIN_NoC_r.inter_icnt_pop_mem_full(_mid)) //arrive
                                    KAIN_NoC_r.inter_icnt_pop_mem_push(mf, _mid);
                                else if (i != mf->get_chip_id()/8 && !KAIN_NoC_r.forward_waiting_full(i))//forward
                                    KAIN_NoC_r.forward_waiting_push(mf, i);
                    }
                }
            }
        //	printf("ZSQ: leave SM_SIDE_LLC == 1 C\n");
#endif

#if SM_SIDE_LLC == 0
        for (unsigned i = 0; i < 4; i++) {
            mem_fetch *mf = (mem_fetch *) ::icnt_pop(192 + i);
#if BEN_OUTPUT == 1
            std::ostringstream out1;
            //mf->set_chiplet(i);
#endif
            if (mf != NULL && INTER_TOPO == 0) { //ZSQ0126, 0 for full connection
                unsigned _cid = mf->get_sid();
                unsigned _subid = mf->get_sub_partition_id();
                unsigned response_size = mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
                if (mf->get_chip_id() / 8 != i && !KAIN_NoC_r.inter_icnt_pop_sm_full(_cid)) { //reply, will push to cluster m_response_fifo
                    KAIN_NoC_r.inter_icnt_pop_sm_push(mf, _cid);
                    mf->set_chiplet(mf->get_sid()/32);
#if BEN_OUTPUT == 1
                    if(gpu_sim_cycle >= 1000000) {
                        out1 << "SM push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                             "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type() << "\tcycle: " <<
                             ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize: " << response_size << "\n";
                        rep3->apply(out1.str().c_str());
                    }
#endif
                }
                else if (mf->get_chip_id() / 8 == i && !KAIN_NoC_r.inter_icnt_pop_llc_full(_subid)) { //request, will push to LLC
                    KAIN_NoC_r.inter_icnt_pop_llc_push(mf, _subid);
#if BEN_OUTPUT == 1
                    if(gpu_sim_cycle >= 1000000) {
                        out1 << "icnt_llc_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                             "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                             << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << i << "\tsize: " << response_size << "\n";
                        rep3->apply(out1.str().c_str());
                    }
#endif
                }
            }
            else if (mf != NULL && INTER_TOPO == 1) { //ZSQ0126, 1 for ring, forwarding if not neighbor
                unsigned _cid = mf->get_sid();
                unsigned _subid = mf->get_sub_partition_id();
                unsigned temp_size;
                mf->set_chiplet(i);
                if (mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK) { //reply
                    if(mf->get_type() == READ_REPLY){
                        temp_size = mf->size();
                    }
                    else if(mf->get_type() == WRITE_ACK){
                        temp_size = mf->get_ctrl_size();
                    }
                    if (i == mf->get_sid() / 32 && !KAIN_NoC_r.inter_icnt_pop_sm_full(_cid)) { //arrive
                        KAIN_NoC_r.inter_icnt_pop_sm_push(mf, _cid);
#if BEN_OUTPUT == 1
                        mf->set_chiplet(mf->get_sid()/32);
                        if(gpu_sim_cycle > 1000000) {
                            out1 << "SM push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                 "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type() << "\tcycle: " <<
                                 ::_get_icnt_cycle() << "\tchip: " << mf->get_chiplet() << "\tsize: " << temp_size << "\n";
                            rep3->apply(out1.str().c_str());
                        }
#endif
                    }
                    else if (i != mf->get_sid() / 32 && !KAIN_NoC_r.forward_waiting_full(i)) {//forward
                        KAIN_NoC_r.forward_waiting_push(mf, i);
#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle > 1000000) {
                            out1 << "FW push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                 "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                 << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << i << "\tsize: " << temp_size
                                 << "\n";
                            rep3->apply(out1.str().c_str());
                        }
#endif
                    }
                }
                else { //request
                    unsigned temp_size;
                    if(mf->get_type() == READ_REQUEST){
                        temp_size = mf->get_ctrl_size();
                    }
                    else if(mf->get_type() == WRITE_REQUEST){
                        temp_size = mf->size();
                    }
                    if (i == mf->get_chip_id() / 8 && !KAIN_NoC_r.inter_icnt_pop_llc_full(_subid)) { //arrive
                        KAIN_NoC_r.inter_icnt_pop_llc_push(mf, _subid);
#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle >= 1000000) {
                            out1 << "icnt_llc_push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                 "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                 << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << i << "\tsize: " << temp_size << "\n";
                            rep3->apply(out1.str().c_str());
                        }
#endif
                    }
                    else if (i != mf->get_chip_id() / 8 && !KAIN_NoC_r.forward_waiting_full(i)) {//forward
                        KAIN_NoC_r.forward_waiting_push(mf, i);
#if BEN_OUTPUT == 1
                        if(gpu_sim_cycle >= 1000000) {
                            out1 << "FW push\tsrc: " << mf->get_src() << "\tdst: " << mf->get_dst() <<
                                 "\tID: " << mf->get_request_uid() << "\ttype: " << mf->get_type()
                                 << "\tcycle: " << ::_get_icnt_cycle() << "\tchip: " << i << "\tsize: " << temp_size << "\n";
                            rep3->apply(out1.str().c_str());
                        }
#endif
                    }
                }
            }
        }
#endif
    }

    if (clock_mask & CHIPLET) {
        static long long kain_chiplet_cycle = 0;
        kain_chiplet_cycle++;

        KAIN_NoC_r.Chiplet_cycle_near_n();
        KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();
//	KAIN_NoC_r.Chiplet_cycle_near_r();

        KAIN_NoC_r.Chiplet_cycle_near_internal();
        KAIN_NoC_r.Chiplet_cycle_near_internal();
        //if (!(kain_chiplet_cycle % 2))
        // KAIN_NoC_r.Chiplet_cycle_near();
        //if (!(kain_chiplet_cycle % 4)) 
        //if (!(kain_chiplet_cycle % 64))  
        //if (!(kain_chiplet_cycle % 32)) 
        //if (!(kain_chiplet_cycle % 8)) //64GB per direction
        //if (!(kain_chiplet_cycle % 2))
        //if (!(kain_chiplet_cycle % 16)) //32GB per direction per link
        //512GB per direction
        KAIN_NoC_r.Chiplet_cycle_remote();
        //KAIN_NoC_r.Chiplet_cycle_remote();
        //KAIN_NoC_r.Chiplet_cycle_remote();
        //KAIN_NoC_r.Chiplet_cycle_remote();
//	KAIN_NoC_r.Chiplet_cycle_remote();
//        KAIN_NoC_r.Chiplet_cycle_remote();
//        KAIN_NoC_r.Chiplet_cycle_remote();
//        KAIN_NoC_r.Chiplet_cycle_remote();
    }
}

void shader_core_ctx::dump_warp_state( FILE *fout ) const
{
   fprintf(fout, "\n");
   fprintf(fout, "per warp functional simulation status:\n");
   for (unsigned w=0; w < m_config->max_warps_per_shader_kain(m_sid); w++ )
       m_warp[w].print(fout);
}

void gpgpu_sim::dump_pipeline( int mask, int s, int m ) const
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(i,stdout,1,mask & 0x2E);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         //if(mask&0x100000) m_memory_partition_unit[i]->print_stat(stdout);
         //if(mask&0x1000000)   m_memory_partition_unit[i]->visualize();
         //if(mask&0x10000000)   m_memory_partition_unit[i]->print(stdout);
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

const struct shader_core_config * gpgpu_sim::getShaderCoreConfig()
{
   return m_shader_config;
}

const struct memory_config * gpgpu_sim::getMemoryConfig()
{
   return m_memory_config;
}

simt_core_cluster * gpgpu_sim::getSIMTCluster()
{
   return *m_cluster;
}

void gpgpu_sim::set_mk_scheduler(MKScheduler* mk_sched)
{
  assert(mk_sched != NULL);
  scheduler = mk_sched;
  for (unsigned i=0; i < m_memory_config->m_n_mem_sub_partition; ++i) {
    m_memory_sub_partition[i]->set_mk_scheduler(mk_sched);
  }

  for (unsigned i=0; i < m_shader_config->n_simt_clusters; ++i) {
    m_cluster[i]->set_mk_scheduler(mk_sched);
  }

  SchedulerUpdateInfo info;
  info.add("Setup", 0);
  info.add("NumSubPartition", m_memory_config->m_n_mem_sub_partition);
  info.add("NumMSHR", m_memory_config->m_L2_config.get_mshr_entries());
  scheduler->update_scheduler(info);
}

void gpgpu_sim::inc_simulated_insts_for_SM(unsigned sid, unsigned num_insts)
{
  const unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
  const unsigned core_id    = m_shader_config->sid_to_cid(sid);

  m_cluster[cluster_id]->get_core(core_id)->get_kernel()->inc_num_simulated_insts(num_insts);
}

bool gpgpu_sim::has_context_switching_core() const
{
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
        for (unsigned j = 0; j < m_shader_config->n_simt_cores_per_cluster; ++j) {
            if (m_cluster[i]->get_core(j)->is_switching()) {
                return true;
            }
        }
    }
    return false;
}

// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
