// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
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

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "option_parser.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "stream_manager.h"

#define MAX(a,b) (((a)>(b))?(a):(b))

time_t g_simulation_starttime;

gpgpu_sim_config g_the_gpu_config;
gpgpu_sim *g_the_gpu;
stream_manager *g_stream_manager;

static int sg_argc = 3;
const char *sg_argv[] = {"", "-config","gpgpusim.config"};

static void print_simulation_time();


bool KAIN_reset_cycle = false;
 
kernel_info_t*
gpgpu_sim_progress(LauncherOptionParser* opt)
{
    kernel_info_t *finished_kernel = NULL;
    while (g_the_gpu->active()) {
        // if any one of the kernel finishes, stop simulation, and come back with new kernel
        // this should be checked before cycle(), because multiple kernels can be finished at the same cycle
        if ((finished_kernel = g_stream_manager->check_finished_kernel()) != NULL) {
            break;
        }

        g_the_gpu->cycle();
        g_the_gpu->deadlock_check();


        if (KAIN_reset_cycle == true) {
            KAIN_reset_cycle = false;
            printf("KAIN reset cycle From %lld\n", opt->run_until());
            assert((float) get_curr_cycle() < (float) opt->run_until() / 4.0 * 3.0);
            opt->KAIN_reset_cycle((float) opt->run_until() / 4.0 * 3.0);
            printf("KAIN reset cycle To %lld\n", opt->run_until());
        }

        // check if we reached the given simulation cycle
        if (opt->run_until_cycle() && (get_curr_cycle() >= opt->run_until())) {
            break;
        }

        // check if we simulated given number of instructions for all child process
        if (opt->run_until_inst() && opt->has_run_until_inst_finished()) {
            break;
        }
    }

    if (!g_the_gpu->active()) {
        // temporary fix
        // if only one kernel was running, finished_kernel is still NULL
        finished_kernel = g_stream_manager->check_finished_kernel();
        assert(finished_kernel != NULL);
    }

    g_the_gpu->print_stats();
    if (finished_kernel != NULL) {
        g_the_gpu->clear_executed_kernel_info(finished_kernel);
    }

    if (g_debug_execution >= 3) {
        printf("GPGPU-Sim: ** STOP kernel simulation **\n");
        fflush(stdout);
    }

    // do not update so that gpu_tot_sim_cycle stays the same
    //g_the_gpu->update_stats();
    print_simulation_time();

    return finished_kernel;
}

void synchronize()
{
  // without pthread, no longer needed
}

void exit_simulation()
{
  // without pthread, no longer needed
  printf("GPGPU-Sim: Parent process terminating.\n");
  fflush(stdout);
}

extern bool g_cuda_launch_blocking;

std::set <unsigned long long> KAIN_page_addr[4];
gpgpu_sim *gpgpu_ptx_sim_init_perf()
{
   srand(1);
   print_splash();
   read_sim_environment_variables();
   read_parser_environment_variables();
   option_parser_t opp = option_parser_create();

   icnt_reg_options(opp);
   g_the_gpu_config.reg_options(opp); // register GPU microrachitecture options
   ptx_reg_options(opp);
   ptx_opcocde_latency_options(opp);
   option_parser_cmdline(opp, sg_argc, sg_argv); // parse configuration options
   fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
   option_parser_print(opp, stdout);
   // Set the Numeric locale to a standard locale where a decimal point is a "dot" not a "comma"
   // so it does the parsing correctly independent of the system environment variables
   assert(setlocale(LC_NUMERIC,"C"));
   g_the_gpu_config.init();

   g_the_gpu = new gpgpu_sim(g_the_gpu_config);
   g_stream_manager = new stream_manager(g_the_gpu,g_cuda_launch_blocking);

   g_simulation_starttime = time((time_t *)NULL);


    //KAIN here, we need to read the page trace
   {
        std::ifstream fchip0("chip0");
        std::ifstream fchip1("chip1");
        std::ifstream fchip2("chip2");
        std::ifstream fchip3("chip3");

        if(!fchip0 || !fchip1 || !fchip2 || !fchip3)
        {
            printf("open chip0-1-2-3 failed\n");
            fflush(stdout);
            assert(0);
        }
        else
        {
            std::string tmp;
            while(getline(fchip0,tmp))
            {
                unsigned long long llNum;
                llNum = std::strtoull(tmp.c_str(), 0, 16);
                KAIN_page_addr[0].insert(llNum);
            }

            while(getline(fchip1,tmp))
            {
                unsigned long long llNum;
                llNum = std::strtoull(tmp.c_str(), 0, 16);
                KAIN_page_addr[1].insert(llNum);
            }
            while(getline(fchip2,tmp))
            {
                unsigned long long llNum;
                llNum = std::strtoull(tmp.c_str(), 0, 16);
                KAIN_page_addr[2].insert(llNum);
            }

            while(getline(fchip3,tmp))
            {
                unsigned long long llNum;
                llNum = std::strtoull(tmp.c_str(), 0, 16);
                KAIN_page_addr[3].insert(llNum);
            }
        }
    }

   return g_the_gpu;
}

void start_sim_thread(int api)
{
  // without pthread, no longer needed
  // gpu no longer measures per kernel statistics
  g_the_gpu->init();
}

void print_simulation_time()
{
   time_t current_time, difference, d, h, m, s;
   current_time = time((time_t *)NULL);
   difference = MAX(current_time - g_simulation_starttime, 1);

   d = difference/(3600*24);
   h = difference/3600 - 24*d;
   m = difference/60 - 60*(h + 24*d);
   s = difference - 60*(m + 60*(h + 24*d));

   fflush(stderr);
   printf("\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
          (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference );
   printf("gpgpu_simulation_rate = %u (inst/sec)\n", (unsigned)(g_the_gpu->gpu_tot_sim_insn / difference) );
   printf("gpgpu_simulation_rate = %u (cycle/sec)\n", (unsigned)(gpu_tot_sim_cycle / difference) );
   fflush(stdout);
}

unsigned long long
get_curr_cycle()
{
  return gpu_sim_cycle + gpu_tot_sim_cycle;
}

