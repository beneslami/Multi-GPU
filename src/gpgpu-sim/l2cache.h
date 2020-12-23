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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../../config.h"
//#include "dram.h"
#include "../abstract_hardware_model.h"
#include "../ramulator_sim/gpu_wrapper.h"

#include <list>
#include <queue>
#include <zlib.h>
#include <set>

#include "../../launcher/mk-sched/mk_scheduler.h"

///////////////////////////////////added by shiqing to implement l1.5
extern std::list<mem_fetch*> remote_cache_request[CHIPLET_NUM];
extern std::list<mem_fetch*> remote_cache_reply[CHIPLET_NUM];
extern std::list<mem_fetch*> KAIN_Remote_Memory_request[32]; // default memory partition number 32
extern new_addr_type *kain_cache[4];
///////////////////////////////////added by shiqing
extern unsigned long long remote_cache_read_hit[4];
extern unsigned long long remote_cache_write_hit[4];
extern unsigned long long remote_cache_read_miss[4];
extern unsigned long long remote_cache_write_miss[4];
extern unsigned long long remote_cache_read_reply[4];


extern unsigned long long KAIN_request_Near;
extern unsigned long long KAIN_request_Remote;
extern unsigned long long KAIN_reply_Near;
extern unsigned long long KAIN_reply_Remote;

class KAIN_GPU_chiplet 
{
    public:
        KAIN_GPU_chiplet(unsigned MC_count)
        {
            assert(MC_count<4);

            for(int i = 0; i < 32; i++)
            {
               Request[i] = new fifo_pipeline<mem_fetch>("Request_",0,256);  
               Reply[i] = new fifo_pipeline<mem_fetch>("Reply_",0,256);  
	       //////////////////////////////////added by shiqing
	       #if DECOUPLE_NEAR_REMOTE == 1
               Request_n[i] = new fifo_pipeline<mem_fetch>("Request_n_",0,256);  
               Request_r[i] = new fifo_pipeline<mem_fetch>("Request_r_",0,256);  
               Reply_n[i] = new fifo_pipeline<mem_fetch>("Reply_n_",0,256);  
               Reply_r[i] = new fifo_pipeline<mem_fetch>("Reply_r_",0,256);  
	       req_turn[i] = 0;
	       rep_turn[i] = 0;
	       #endif
            }
            for(int i = 0; i < 4; i++)
            {
               Request_Near[i] = new fifo_pipeline<mem_fetch>("Request_Near_",0,256);  
               Request_Remote[i] = new fifo_pipeline<mem_fetch>("Request_Remote_",0,256);  

               Reply_Near[i] = new fifo_pipeline<mem_fetch>("Reply_Near_",0,256);  
               Reply_Remote[i] = new fifo_pipeline<mem_fetch>("Reply_Remote_",0,256);  

               Request_turn[i] = 0;
               Reply_turn[i] = 0;
               Last_Remote_ID_ID[i] = 0;
               Last_Near_ID_ID[i] = 0;
	      
                for(int j = 0; j < 4; j++)
                {
                    Request_Remote_Src_From[i][j] = new fifo_pipeline<mem_fetch>("Request_Remote_",0,256);
                    Reply_Remote_Src_From[i][j] = new fifo_pipeline<mem_fetch>("Reply_Remote_",0,256);
                    Request_Near_Src_From[i][j] = new fifo_pipeline<mem_fetch>("Request_Near_",0,256);
                    Reply_Near_Src_From[i][j] = new fifo_pipeline<mem_fetch>("Reply_Near_",0,256);
                    Remote_Request_turn[i][j] = 0;
                    Near_Request_turn[i][j] = 0;
                } 

            }

            LastTime_Request = 0;
            LastTime_Reply = 0;
            Remote_cycle = 0;
            Near_cycle = 0;
            Last_Remote_ID = 0;
        }
        void Chiplet_cycle_near()//Near bandwidth is 128byte * Operating frequency
        {
            for(int i = 0; i < 4; i++)
            {
                if(Request_turn[i] == 0)
                {
                    Request_turn[i] = 1;
                    if(!Request_Near[i]->empty())
                    {
                        mem_fetch *mf = Request_Near[i]->top();
                        if(!Request[mf->get_chip_id()]->full())
                        {
                            Request[mf->get_chip_id()]->push(mf);
                            KAIN_request_Near++;
                            Request_Near[i]->pop();
                            //////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Request_Near[%d] -> Request[%d]\n", i, mf->get_chip_id());
                            fflush(stdout);
                            continue;
                        }
                    }
                    if(!Request_Remote[i]->empty())
                    {
                        mem_fetch *mf = Request_Remote[i]->top();
                        #if INTER_DIE_TOPOLOGY == 1
			//////////////////////////////add by shiqing start
			if (mf->get_chip_id()/8 != i) // not neignbor forward
			{
			    if (!Request_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
			    {
				Request_Remote[mf->get_chip_id()/8]->push(mf);
				/////////////////////shiqing print
				printf("ZSQ: Chiplet_cycle_near(), not neignbor forward, Request_Remote[%d] -> Request_Remote[%d]\n", i, mf->get_chip_id()/8);
			        fflush(stdout);
				Request_Remote[i]->pop();
				continue;
			    } 
			} else {
			//////////////////////////////add by shiqing end
			#endif
			if(!Request[mf->get_chip_id()]->full())
                        {
                            Request[mf->get_chip_id()]->push(mf);
                            KAIN_request_Remote++;
                            //printf("mf id %u, TPC %d, mid %d, Write %d\n", mf->get_request_uid(),mf->get_tpc(), mf->get_chip_id(), mf->get_is_write());
                            //fflush(stdout);
			    //////////////////shiqing print
			    printf("ZSQ: Chiplet_cycle_near(), Request_Remote[%d] -> Request[%d]\n", i, mf->get_chip_id());
			    fflush(stdout);
                            Request_Remote[i]->pop();
                        }     
			#if INTER_DIE_TOPOLOGY == 1
				} //else end add by shiqing
                    	#endif
		    }
                }
                else
                {
                    Request_turn[i] = 0;
                    if(!Request_Remote[i]->empty())
                    {
                        mem_fetch *mf = Request_Remote[i]->top();
                        #if INTER_DIE_TOPOLOGY == 1
			//////////////////////////////add by shiqing start
                        if (mf->get_chip_id()/8 != i) // not neignbor forward
                        {    
                            if (!Request_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                            {
                                Request_Remote[mf->get_chip_id()/8]->push(mf);
                                Request_Remote[i]->pop();
				/////////////////////shiqing print
                                printf("ZSQ: Chiplet_cycle_near(), not neignbor forward, Request_Remote[%d] -> Request_Remote[%d]\n", i, mf->get_chip_id()/8);
                                fflush(stdout);
				continue;
                            }
                        } else {
                        //////////////////////////////add by shiqing end
			#endif
			if(!Request[mf->get_chip_id()]->full())
                        {
                            Request[mf->get_chip_id()]->push(mf);
                            KAIN_request_Remote++;
                            //printf("mf id %u, TPC %d, mid %d, Write %d\n", mf->get_request_uid(), mf->get_tpc(), mf->get_chip_id(), mf->get_is_write());
                            //fflush(stdout);
                            Request_Remote[i]->pop();
                            //////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Request_Remote[%d] -> Request[%d]\n", i, mf->get_chip_id());
                            fflush(stdout);
			    continue;
                        }     
			#if INTER_DIE_TOPOLOGY == 1
				} //else end add by shiqing
                    	#endif
		    }
                    if(!Request_Near[i]->empty())
                    {
                        mem_fetch *mf = Request_Near[i]->top();
                        if(!Request[mf->get_chip_id()]->full())
                        {
                            Request[mf->get_chip_id()]->push(mf);
                            KAIN_request_Near++;
                            Request_Near[i]->pop();
                            //////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Request_Near[%d] -> Request[%d]\n", i, mf->get_chip_id());
                            fflush(stdout);
                        }
                    }
                }
            }


            for(int i = 0; i < 4; i++)
            {
                if(Reply_turn[i] == 0)
                {
                    Reply_turn[i] = 1;
                    if(!Reply_Near[i]->empty())
                    {
                        mem_fetch *mf = Reply_Near[i]->top();
			if(!Reply[mf->get_sub_partition_id()/2]->full())
                        {
                            Reply[mf->get_sub_partition_id()/2]->push(mf);
                            KAIN_reply_Near++;
                            Reply_Near[i]->pop();
                            //////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Reply_Near[%d] -> Reply[%d]\n", i, mf->get_sub_partition_id()/2);
                            fflush(stdout);
			    continue;
                        }  
                    }
                    if(!Reply_Remote[i]->empty())
                    {
                        mem_fetch *mf = Reply_Remote[i]->top();
                        #if INTER_DIE_TOPOLOGY == 1
			//////////////////////////////add by shiqing start
                        if (mf->get_chip_id()/8 != i) // not neignbor forward
                        {
                            if (!Reply_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                            {
                                Reply_Remote[mf->get_chip_id()/8]->push(mf);
                                Reply_Remote[i]->pop();
                                /////////////////////shiqing print
                                printf("ZSQ: Chiplet_cycle_near(), not neignbor forward, Reply_Remote[%d] -> Reply_Remote[%d]\n", i, mf->get_chip_id()/8);
                                fflush(stdout);
                                continue;
                            }
                        } else {
                        //////////////////////////////add by shiqing end
			#endif
			if(!Reply[mf->get_sub_partition_id()/2]->full())
                        {
                            Reply[mf->get_sub_partition_id()/2]->push(mf);
                            KAIN_reply_Remote++;
                            Reply_Remote[i]->pop();
			    ///////////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Reply_Remote[%d] -> Reply[%d]\n", i, mf->get_sub_partition_id()/2);
                            fflush(stdout);
                        }     
			#if INTER_DIE_TOPOLOGY == 1
				} //else end add by shiqing
                   	#endif
		    }
                }
                else
                {
                    Reply_turn[i] = 0;
                    if(!Reply_Remote[i]->empty())
                    {
                        mem_fetch *mf = Reply_Remote[i]->top();
                        #if INTER_DIE_TOPOLOGY == 1
			//////////////////////////////add by shiqing start
                        if (mf->get_chip_id()/8 != i) // not neignbor forward
                        {
                            if (!Reply_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                            {
                                Reply_Remote[mf->get_chip_id()/8]->push(mf);
                                Reply_Remote[i]->pop();
                                /////////////////////shiqing print
                                printf("ZSQ: Chiplet_cycle_near(), not neignbor forward, Reply_Remote[%d] -> Reply_Remote[%d]\n", i, mf->get_chip_id()/8);
                                fflush(stdout);
                                continue;
                            }
                        } else {
                        //////////////////////////////add by shiqing end
			#endif
			if(!Reply[mf->get_sub_partition_id()/2]->full())
                        {
                            Reply[mf->get_sub_partition_id()/2]->push(mf);
                            KAIN_reply_Remote++;
                            Reply_Remote[i]->pop();
			    ///////////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Reply_Remote[%d] -> Reply[%d]\n", i, mf->get_sub_partition_id()/2);
                            fflush(stdout);
                            continue;
                        }     
			#if INTER_DIE_TOPOLOGY == 1
				} //else end add by shiqing
                    	#endif
		    }
                    if(!Reply_Near[i]->empty())
                    { 
                        mem_fetch *mf = Reply_Near[i]->top();
                        if(!Reply[mf->get_sub_partition_id()/2]->full())
                        {
                            Reply[mf->get_sub_partition_id()/2]->push(mf);
                            KAIN_reply_Near++;
                            Reply_Near[i]->pop();
                            ///////////////////////shiqing print
                            printf("ZSQ: Chiplet_cycle_near(), Reply_Near[%d] -> Reply[%d]\n", i, mf->get_sub_partition_id()/2);
                            fflush(stdout);
                        }
                    }
                }
            }


        }

///////////////////////////////////////////add by shiqing start
#if DECOUPLE_NEAR_REMOTE == 0
	void Chiplet_cycle_near_n()
	{
            for (int i = 0; i < 4; i++)
            {
                if(!Reply_Near[i]->empty())
                {
                    mem_fetch *mf = Reply_Near[i]->top();
                    if(!Reply[mf->get_sub_partition_id()/2]->full())
                    {
                         Reply[mf->get_sub_partition_id()/2]->push(mf);
                         KAIN_reply_Near++;
                         Reply_Near[i]->pop();  
			 //printf("ZSQ: Chiplet_cycle_near_n(), Reply_Near[%d] -> Reply[%d]. ", i, mf->get_sub_partition_id()/2);
			 //mf->mf_print();
			 fflush(stdout);
                    }
                }
            }

	    for (int i = 0; i < 4; i++)
	    {
    		if(!Request_Near[i]->empty())
		{
		    mem_fetch *mf = Request_Near[i]->top();
		    if(!Request[mf->get_chip_id()]->full())
		    {
			 Request[mf->get_chip_id()]->push(mf);
                         KAIN_request_Near++;
                         Request_Near[i]->pop();		    
			 //printf("ZSQ: Chiplet_cycle_near_n(), Request_Near[%d] -> Request[%d]. ", i, mf->get_chip_id());
			 //mf->mf_print();
                         fflush(stdout);
		    }
		}
	    }
	}
        void Chiplet_cycle_near_r()
        {
            for (int i = 0; i < 4; i++)
            {
                if(!Reply_Remote[i]->empty())
                {
                    mem_fetch *mf = Reply_Remote[i]->top();
                    #if INTER_DIE_TOPOLOGY == 1
                    //////////////////////////////add by shiqing start
                    if (mf->get_chip_id()/8 != i) // not neignbor forward
                    {
                        if (!Reply_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                        {
                            Reply_Remote[mf->get_chip_id()/8]->push(mf);
                            Reply_Remote[i]->pop();
			    //printf("ZSQ: Chiplet_cycle_near_r(), not neignbor forward, Reply_Remote[%d] -> Reply_Remote[%d]. ", i, mf->get_chip_id()/8);
			    //mf->mf_print();
                            fflush(stdout);
                            continue;
                        }
                    } else {
                    //////////////////////////////add by shiqing end
                    #endif                       
                    if(!Reply[mf->get_sub_partition_id()/2]->full())
                    {
                        Reply[mf->get_sub_partition_id()/2]->push(mf);
                        KAIN_reply_Remote++;
                        //printf("mf id %u, TPC %d, mid %d, Write %d\n", mf->get_request_uid(),mf->get_tpc(), mf->get_chip_id(), mf->get_is_write());
                        //fflush(stdout);
                        Reply_Remote[i]->pop();
                        //printf("ZSQ: Chiplet_cycle_near_r(), Reply_Remote[%d] -> Reply[%d]. ", i, mf->get_sub_partition_id()/2);
			//mf->mf_print();
                        fflush(stdout);
                    }
                    #if INTER_DIE_TOPOLOGY == 1
                            } //else end add by shiqing
                    #endif
                 }
            } 
	    for (int i = 0; i < 4; i++)
            {
                if(!Request_Remote[i]->empty())
                {
                    mem_fetch *mf = Request_Remote[i]->top();
                    #if INTER_DIE_TOPOLOGY == 1
                    //////////////////////////////add by shiqing start
                    if (mf->get_chip_id()/8 != i) // not neignbor forward
                    {
                        if (!Request_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                        {
                            Request_Remote[mf->get_chip_id()/8]->push(mf);
                            Request_Remote[i]->pop();
                            //printf("ZSQ: Chiplet_cycle_near_r(), not neignbor forward, Request_Remote[%d] -> Request_Remote[%d]. ", i, mf->get_chip_id()/8);
			    //mf->mf_print();
                            fflush(stdout);
                            continue;
                        }
                    } else {
                    //////////////////////////////add by shiqing end
                    #endif
                    if(!Request[mf->get_chip_id()]->full())
                    {
                        Request[mf->get_chip_id()]->push(mf);
                        KAIN_request_Remote++;
                        //printf("ZSQ: KAIN_request_Remote++, mf id %u, TPC %d, mid %d, Write %d\n", mf->get_request_uid(),mf->get_tpc(), mf->get_chip_id(), mf->get_is_write());
                        if (mf->get_access_type() == INST_ACC_R) { 
				//printf("ZSQ: Inst request miss, inst @ pc=0x%04x\n", mf->get_pc());
			}
			fflush(stdout);
                        Request_Remote[i]->pop();
                        //printf("ZSQ: Chiplet_cycle_near_r(), Request_Remote[%d] -> Request[%d]. ", i, mf->get_chip_id());
			//mf->mf_print();
                        fflush(stdout);
		    }
                    #if INTER_DIE_TOPOLOGY == 1
                            } //else end add by shiqing
                    #endif
                 }
	    }            
        }
#endif

#if DECOUPLE_NEAR_REMOTE == 1
        void Chiplet_cycle_near_n()
        {
            for (int i = 0; i < 4; i++)
            {
                if(!Reply_Near[i]->empty())
                {
                    mem_fetch *mf = Reply_Near[i]->top();
                    if(!Reply_n[mf->get_sub_partition_id()/2]->full())
                    {
                         Reply_n[mf->get_sub_partition_id()/2]->push(mf);
                         KAIN_reply_Near++;
                         Reply_Near[i]->pop();
                         //printf("ZSQ: Chiplet_cycle_near_n(), Reply_Near[%d] -> Reply_n[%d]. ", i, mf->get_sub_partition_id()/2);
			 //mf->mf_print();
                         fflush(stdout);
                    }
                }
            }

            for (int i = 0; i < 4; i++)
            {
                if(!Request_Near[i]->empty())
                {
                    mem_fetch *mf = Request_Near[i]->top();
                    if(!Request_n[mf->get_chip_id()]->full())
                    {
                         Request_n[mf->get_chip_id()]->push(mf);
                         KAIN_request_Near++;
                         Request_Near[i]->pop();
                         //printf("ZSQ: Chiplet_cycle_near_n(), Request_Near[%d] -> Request_n[%d]. ", i, mf->get_chip_id());
			 //mf->mf_print();
                         fflush(stdout);
                    }
                }
            }
        }
        void Chiplet_cycle_near_r()
        {
            for (int i = 0; i < 4; i++)
            {
                if(!Reply_Remote[i]->empty())
                {
                    mem_fetch *mf = Reply_Remote[i]->top();
                    #if INTER_DIE_TOPOLOGY == 1
                    //////////////////////////////add by shiqing start
                    if (mf->get_chip_id()/8 != i) // not neignbor forward
                    {
                        if (!Reply_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                        {
                            Reply_Remote[mf->get_chip_id()/8]->push(mf);
                            Reply_Remote[i]->pop();
                            //printf("ZSQ: Chiplet_cycle_near_r(), not neignbor forward, Reply_Remote[%d] -> Reply_Remote[%d]. ", i, mf->get_chip_id()/8);
			    //mf->mf_print();
                            fflush(stdout);
                            continue;
                        }
                    } else {
                    //////////////////////////////add by shiqing end
                    #endif
                    if(!Reply_r[mf->get_sub_partition_id()/2]->full())
                    {
                        Reply_r[mf->get_sub_partition_id()/2]->push(mf);
                        KAIN_reply_Remote++;
                        //printf("mf id %u, TPC %d, mid %d, Write %d\n", mf->get_request_uid(),mf->get_tpc(), mf->get_chip_id(), mf->get_is_write());
                        //fflush(stdout);
                        Reply_Remote[i]->pop();
                        //printf("ZSQ: Chiplet_cycle_near_r(), Reply_Remote[%d] -> Reply_r[%d]. ", i, mf->get_sub_partition_id()/2);
			//mf->mf_print();
                        fflush(stdout);
                    }
                    #if INTER_DIE_TOPOLOGY == 1
                            } //else end add by shiqing
                    #endif
                 }
            }
            for (int i = 0; i < 4; i++)
            {
                if(!Request_Remote[i]->empty())
                {
                    mem_fetch *mf = Request_Remote[i]->top();
                    #if INTER_DIE_TOPOLOGY == 1
                    //////////////////////////////add by shiqing start
                    if (mf->get_chip_id()/8 != i) // not neignbor forward
                    {
                        if (!Request_Remote[mf->get_chip_id()/8]->full()) // real data location chiplet
                        {
                            Request_Remote[mf->get_chip_id()/8]->push(mf);
                            Request_Remote[i]->pop();
                            //printf("ZSQ: Chiplet_cycle_near_r(), not neignbor forward, Request_Remote[%d] -> Request_Remote[%d]. ", i, mf->get_chip_id()/8);
			    //mf->mf_print();
                            fflush(stdout);
                            continue;
                        }
                    } else {
                    //////////////////////////////add by shiqing end
                    #endif
                    if(!Request_r[mf->get_chip_id()]->full())
                    {
                        Request_r[mf->get_chip_id()]->push(mf);
                        KAIN_request_Remote++;
                        //printf("ZSQ: KAIN_request_Remote++, mf id %u, TPC %d, mid %d, Write %d\n", mf->get_request_uid(),mf->get_tpc(), mf->get_chip_id(), mf->get_is_write());
                        //if(mf->get_access_type() == INST_ACC_R) printf("ZSQ: Inst request miss, inst @ pc=0x%04x\n", mf->get_pc());
			fflush(stdout);
                        Request_Remote[i]->pop();
                        //printf("ZSQ: Chiplet_cycle_near_r(), Request_Remote[%d] -> Request_r[%d]. ", i, mf->get_chip_id());
			//mf->mf_print();
                        fflush(stdout);
                    }
                    #if INTER_DIE_TOPOLOGY == 1
                            } //else end add by shiqing
                    #endif
                 }
            }
        }
#endif
/////////////////////////////////////////////////add by shiqing end

        //Request_Remote_Src_From[8][8];//8 is Src, 8 is from
        void Chiplet_cycle_remote()//Remote bandwidth is 128byte * Operating frequency
        {
            for(int mm = 0; mm < 4; mm++)
            {
                int i = mm;
                int Last_Remote_ID_ID_tmp = Last_Remote_ID_ID[i];
                for(int j = 0; j < 4; j++)//from j
                {
                    //int ii = (j + Remote_cycle) % 4;
                    int ii = (j + Last_Remote_ID_ID_tmp) % 4;

                    if(Remote_Request_turn[i][ii] == 1)
                    {
		        Remote_Request_turn[i][ii] = 0;
			#if INTER_DIE_TOPOLOGY == 1
			/////////////////////////////add by shiqing start
                        if (i%2 == ii%2) //not neighbor
                        {
			    if (!Request_Remote[(ii+1)%4]->full() && !Request_Remote_Src_From[i][ii]->empty())
			    {
				mem_fetch *mf = Request_Remote_Src_From[i][ii]->top();
				Request_Remote[(ii+1)%4]->push(mf);
				Request_Remote_Src_From[i][ii]->pop();
                        	Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Request_Remote_Src_From[%d][%d] -> Request_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii+1)%4, i, ii+1);
			        //mf->mf_print();
				fflush(stdout);
                        	continue;
			    }
			    else if (!Request_Remote[(ii-1+4)%4]->full() && !Request_Remote_Src_From[i][ii]->empty())				       {   
                                mem_fetch *mf = Request_Remote_Src_From[i][ii]->top();
                                Request_Remote[(ii-1+4)%4]->push(mf);
                                Request_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Request_Remote_Src_From[%d][%d] -> Request_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii-1+4)%4, i, ii+1);
			        //mf->mf_print();
                                fflush(stdout);
                                continue;
                            }    
                            if (!Reply_Remote[(ii+1)%4]->full() && !Reply_Remote_Src_From[i][ii]->empty())
                            {   
                                mem_fetch *mf = Reply_Remote_Src_From[i][ii]->top();
                                Reply_Remote[(ii+1)%4]->push(mf);
                                Reply_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Reply_Remote_Src_From[%d][%d] -> Reply_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii+1)%4, i, ii+1);
			 	//mf->mf_print();
                                fflush(stdout);
                                continue;
                            }
                            else if (!Reply_Remote[(ii-1+4)%4]->full() && !Reply_Remote_Src_From[i][ii]->empty())                                      {
                                mem_fetch *mf = Reply_Remote_Src_From[i][ii]->top();
                                Reply_Remote[(ii-1+4)%4]->push(mf);
                                Reply_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Reply_Remote_Src_From[%d][%d] -> Reply_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii-1+4)%4, i, ii+1);
			 	//mf->mf_print();
                                fflush(stdout);
                                continue;
                            }
             		} else {
                        /////////////////////////////add by shiqing end
			#endif
			if(!Request_Remote[i]->full() && !Request_Remote_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Request_Remote_Src_From[i][ii]->top();
                        Request_Remote[i]->push(mf);
                        Request_Remote_Src_From[i][ii]->pop();
                        Last_Remote_ID_ID[i] = ii+1;
			//printf("ZSQ: Chiplet_cycle_remote(), Request_Remote_Src_From[%d][%d] -> Request_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, i, i, ii+1);
			//mf->mf_print();
			fflush(stdout);
                        continue;
                        }
                        if(!Reply_Remote[i]->full() && !Reply_Remote_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Reply_Remote_Src_From[i][ii]->top();
                        Reply_Remote[i]->push(mf);
                        Reply_Remote_Src_From[i][ii]->pop();
                        Last_Remote_ID_ID[i] = ii+1;
			//printf("ZSQ: Chiplet_cycle_remote(), Reply_Remote_Src_From[%d][%d] -> Reply_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, i, i, ii+1);
			//mf->mf_print();
                        fflush(stdout);
                        }     
			#if INTER_DIE_TOPOLOGY == 1
				}//else end add by shiqing
                    	#endif
		    }
                    else
                    {
                        Remote_Request_turn[i][ii] = 1;
                        #if INTER_DIE_TOPOLOGY == 1
			/////////////////////////////add by shiqing start
                        if (i%2 == ii%2) //not neighbor
                        {
                            if (!Reply_Remote[(ii+1)%4]->full() && !Reply_Remote_Src_From[i][ii]->empty())
                            {
                                mem_fetch *mf = Reply_Remote_Src_From[i][ii]->top();
                                Reply_Remote[(ii+1)%4]->push(mf);
                                Reply_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Reply_Remote_Src_From[%d][%d] -> Reply_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii+1)%4, i, ii+1);
			 	//mf->mf_print();
                                fflush(stdout);
				continue;
                            }
                            else if (!Reply_Remote[(ii-1+4)%4]->full() && !Reply_Remote_Src_From[i][ii]->empty())                                      {
                                mem_fetch *mf = Reply_Remote_Src_From[i][ii]->top();
                                Reply_Remote[(ii-1+4)%4]->push(mf);
                                Reply_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Reply_Remote_Src_From[%d][%d] -> Reply_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii-1+4)%4, i, ii+1);
			 	//mf->mf_print();
                                fflush(stdout);
                                continue;
                            }
                            if (!Request_Remote[(ii+1)%4]->full() && !Request_Remote_Src_From[i][ii]->empty())
                            {
                                mem_fetch *mf = Request_Remote_Src_From[i][ii]->top();
                                Request_Remote[(ii+1)%4]->push(mf);
                                Request_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Request_Remote_Src_From[%d][%d] -> Request_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii+1)%4, i, ii+1);
			 	//mf->mf_print();
                                fflush(stdout);
                                continue;
                            }
                            else if (!Request_Remote[(ii-1+4)%4]->full() && !Request_Remote_Src_From[i][ii]->empty())                                  {
                                mem_fetch *mf = Request_Remote_Src_From[i][ii]->top();
                                Request_Remote[(ii-1+4)%4]->push(mf);
                                Request_Remote_Src_From[i][ii]->pop();
                                Last_Remote_ID_ID[i] = ii+1;
				//printf("ZSQ: Chiplet_cycle_remote(), not neighbor, Request_Remote_Src_From[%d][%d] -> Request_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, (ii-1+4)%4, i, ii+1);
			 	//mf->mf_print();
                                fflush(stdout);
                                continue;
                            } 
			} else {
                        /////////////////////////////add by shiqing end
			#endif
                        if(!Reply_Remote[i]->full() && !Reply_Remote_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Reply_Remote_Src_From[i][ii]->top();
                        Reply_Remote[i]->push(mf);
                        Reply_Remote_Src_From[i][ii]->pop();
                        Last_Remote_ID_ID[i] = ii+1;
			//printf("ZSQ: Chiplet_cycle_remote(), Reply_Remote_Src_From[%d][%d] -> Reply_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, i, i, ii+1);
			//mf->mf_print();
                        fflush(stdout);
			continue;
                        }
                        if(!Request_Remote[i]->full() && !Request_Remote_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Request_Remote_Src_From[i][ii]->top();
                        Request_Remote[i]->push(mf);
                        Request_Remote_Src_From[i][ii]->pop();
                        Last_Remote_ID_ID[i] = ii+1;
                        //printf("ZSQ: Chiplet_cycle_remote(), Request_Remote_Src_From[%d][%d] -> Request_Remote[%d], Last_Remote_ID_ID[%d] = %d. ", i, ii, i, i, ii+1);
			//mf->mf_print();
                        fflush(stdout);
			}    
			#if INTER_DIE_TOPOLOGY == 1
				} //else end add by shiqing
                    	#endif
		    }
                }
            }

            Remote_cycle++;
        }



        void Chiplet_cycle_near_internal()//Near bandwidth is 128byte * Operating frequency
        {
            for(int i = 0; i < 4; i++)
            {
                int Last_Near_ID_ID_tmp = Last_Near_ID_ID[i];
                for(int j = 0; j < 4; j++)//from j
                {
                    //int ii = (j + Near_cycle) % 4;
                    int ii = (j + Last_Near_ID_ID_tmp) % 4;

                    if(Near_Request_turn[i][ii] == 1)
                    {
                        Near_Request_turn[i][ii] = 0;
                        if(!Request_Near[i]->full() && !Request_Near_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Request_Near_Src_From[i][ii]->top();
                        Request_Near[i]->push(mf);
                        Request_Near_Src_From[i][ii]->pop();
                        Last_Near_ID_ID[i] = ii+1;
                        continue;
                        }
                        if(!Reply_Near[i]->full() && !Reply_Near_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Reply_Near_Src_From[i][ii]->top();
                        Reply_Near[i]->push(mf);
                        Reply_Near_Src_From[i][ii]->pop();
                        Last_Near_ID_ID[i] = ii+1;
                        }
                    }
                    else
                    {
                        Near_Request_turn[i][ii] = 1;
                        if(!Reply_Near[i]->full() && !Reply_Near_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Reply_Near_Src_From[i][ii]->top();
                        Reply_Near[i]->push(mf);
                        Reply_Near_Src_From[i][ii]->pop();
                        Last_Near_ID_ID[i] = ii+1;
                        continue;
                        }
                        if(!Request_Near[i]->full() && !Request_Near_Src_From[i][ii]->empty())
                        {
                        mem_fetch *mf = Request_Near_Src_From[i][ii]->top();
                        Request_Near[i]->push(mf);
                        Request_Near_Src_From[i][ii]->pop();
                        Last_Near_ID_ID[i] = ii+1;
                        }
                    }
                }

            }

            Near_cycle++;
        }






        bool request_full(mem_fetch *mf,unsigned to_m_id, unsigned from_m_id)
        {
            assert(mf!=NULL); 
/*            if(mf->get_tpc()<64)
            {
                if(to_m_id < 2)//near 
                {
                    return Request_Near_Src_From[to_m_id][from_m_id]->full(); 
                }
                else//remote
                {
                    return Request_Remote_Src_From[to_m_id][from_m_id]->full(); 
                }
            }
            else
            {
                if(to_m_id >= 2)//near 
                {
                    return Request_Near_Src_From[to_m_id][from_m_id]->full(); 
                }
                else//remote
                {
                    return Request_Remote_Src_From[to_m_id][from_m_id]->full(); 
                }
            }
*/
	   ////////////////////////////add by shiqing start
           if (from_m_id == to_m_id) //near (local)
		   return Request_Near_Src_From[to_m_id][from_m_id]->full();
	   else
		   return Request_Remote_Src_From[to_m_id][from_m_id]->full(); 
	   ////////////////////////////add by shiqing end
        }
        void request_push(mem_fetch *mf,unsigned to_m_id, unsigned from_m_id)
        {
            assert(mf!=NULL); 
/*            if(mf->get_tpc()<64)
            {
                if(to_m_id < 2)//near 
                {
                    Request_Near_Src_From[to_m_id][from_m_id]->push(mf); 
                }
                else//remote
                {
                    Request_Remote_Src_From[to_m_id][from_m_id]->push(mf); 
                    //printf("How is it possible\n");
                    //fflush(stdout);
                }
            }
            else
            {
                if(to_m_id >= 2)//near 
                {   
                    Request_Near_Src_From[to_m_id][from_m_id]->push(mf); 
                }
                else//remote
                {   
                    Request_Remote_Src_From[to_m_id][from_m_id]->push(mf); 
                    //printf("How is it possible\n");
                    //fflush(stdout);
                }   
            }   
*/
	   ////////////////////////////add by shiqing start
           if (from_m_id == to_m_id) //near (local)
                   return Request_Near_Src_From[to_m_id][from_m_id]->push(mf);
	   else
                   return Request_Remote_Src_From[to_m_id][from_m_id]->push(mf);
           ////////////////////////////add by shiqing end
        }

#if DECOUPLE_NEAR_REMOTE == 0
        bool request_empty(unsigned m_id)
        {
            return Request[m_id]->empty(); 
        }

        mem_fetch * request_top(unsigned m_id)
        {
            return Request[m_id]->top();
        }

        mem_fetch * request_pop_front(unsigned m_id)
        {
            return Request[m_id]->pop();
        }
#endif
#if DECOUPLE_NEAR_REMOTE == 1
        bool request_empty(unsigned m_id)
        {
            return (Request_n[m_id]->empty() && Request_r[m_id]->empty()); 
        }

        mem_fetch * request_top(unsigned m_id)
        {   
	    if ((req_turn[m_id] < threshold_turn && (!Request_n[m_id]->empty())) || Request_r[m_id]->empty())
            	return Request_n[m_id]->top();
	    else 
		return Request_r[m_id]->top();
        }

        mem_fetch * request_pop_front(unsigned m_id)
        {
	    if ((req_turn[m_id] < threshold_turn && (!Request_n[m_id]->empty())) || Request_r[m_id]->empty())
	    {
		req_turn[m_id] = (req_turn[m_id] + 1) % bound_turn;
		return Request_n[m_id]->pop();	
            } else {
	    	req_turn[m_id] = (req_turn[m_id] + 1) % bound_turn;
		return Request_r[m_id]->pop();
	    }
	}
#endif


        bool reply_full(mem_fetch *mf,unsigned to_m_id, unsigned from_m_id)
        {
            assert(mf!=NULL);
/*            if(mf->get_tpc()<64)
            {
                if(from_m_id < 2)//near 
                {
                    return Reply_Near_Src_From[to_m_id][from_m_id]->full();
                }
                else//remote
                {
                    return Reply_Remote_Src_From[to_m_id][from_m_id]->full();
                }
            }
            else
            {
                if(from_m_id >= 2)//near 
                {
                    return Reply_Near_Src_From[to_m_id][from_m_id]->full();
                }
                else//remote
                {
                    return Reply_Remote_Src_From[to_m_id][from_m_id]->full();
                }
            }
*/
           ////////////////////////////add by shiqing start
           if (from_m_id == to_m_id) //near (local)
                   return Reply_Near_Src_From[to_m_id][from_m_id]->full();
           else
                   return Reply_Remote_Src_From[to_m_id][from_m_id]->full();
           ////////////////////////////add by shiqing end
        }
        void reply_push(mem_fetch *mf,unsigned to_m_id, unsigned from_m_id)
        {
            assert(mf!=NULL);
/*            if(mf->get_tpc()<64)
            {
                if(from_m_id < 2)//near 
                {
                    Reply_Near_Src_From[to_m_id][from_m_id]->push(mf);
                }
                else//remote
                {
                    Reply_Remote_Src_From[to_m_id][from_m_id]->push(mf);
                }
            }
            else
            {
                if(from_m_id >= 2)//near 
                {  
                    Reply_Near_Src_From[to_m_id][from_m_id]->push(mf);
                }
                else//remote
                {  
                    Reply_Remote_Src_From[to_m_id][from_m_id]->push(mf);
                }  
            }  
*/
           ////////////////////////////add by shiqing start
           if (from_m_id == to_m_id) //near (local)
                   return Reply_Near_Src_From[to_m_id][from_m_id]->push(mf);
           else
                   return Reply_Remote_Src_From[to_m_id][from_m_id]->push(mf);
           ////////////////////////////add by shiqing end  
      }

#if DECOUPLE_NEAR_REMOTE == 0
        mem_fetch * reply_top(unsigned m_id)
        {
            return Reply[m_id]->top();
        }

        mem_fetch * reply_pop_front(unsigned m_id)
        {
            return Reply[m_id]->pop();
        }
#endif
#if DECOUPLE_NEAR_REMOTE == 1
        mem_fetch * reply_top(unsigned m_id)
        {
            if ((rep_turn[m_id] < threshold_turn && (!Reply_n[m_id]->empty())) | Reply_r[m_id]->empty())
                return Reply_n[m_id]->top();
            else
                return Reply_r[m_id]->top();
        }

        mem_fetch * reply_pop_front(unsigned m_id)
        {
            if ((rep_turn[m_id] < threshold_turn && (!Reply_n[m_id]->empty())) | Reply_r[m_id]->empty()) {
            	rep_turn[m_id] = (rep_turn[m_id] + 1) % bound_turn;
                return Reply_n[m_id]->pop();
            } else {
            	rep_turn[m_id] = (rep_turn[m_id] + 1) % bound_turn;
                return Reply_r[m_id]->pop();
            }
	}
#endif

    private:
        fifo_pipeline<mem_fetch> *Request_Near[4];
        fifo_pipeline<mem_fetch> *Request_Remote[4];
        fifo_pipeline<mem_fetch> *Request[32];
        fifo_pipeline<mem_fetch> *Reply_Near[4];
        fifo_pipeline<mem_fetch> *Reply_Remote[4];
        fifo_pipeline<mem_fetch> *Reply[32];

        fifo_pipeline<mem_fetch> *Request_Remote_Src_From[4][4];//8 is Src, 8 is from
        fifo_pipeline<mem_fetch> *Reply_Remote_Src_From[4][4];

        fifo_pipeline<mem_fetch> *Request_Near_Src_From[4][4];//8 is Src, 8 is from
        fifo_pipeline<mem_fetch> *Reply_Near_Src_From[4][4];

        int LastTime_Request;
        int LastTime_Reply;
        int Request_turn[4];
        int Reply_turn[4];
        long long Remote_cycle;
        long long Near_cycle;
        int Remote_Request_turn[4][4];
        int Near_Request_turn[4][4];
        int Last_Remote_ID_ID[4];
        int Last_Near_ID_ID[4];
        int Last_Remote_ID;

	//////////////////////////////////added by shiqing
	#if DECOUPLE_NEAR_REMOTE == 1
        fifo_pipeline<mem_fetch> *Request_n[32];
        fifo_pipeline<mem_fetch> *Request_r[32];
        fifo_pipeline<mem_fetch> *Reply_n[32];
        fifo_pipeline<mem_fetch> *Reply_r[32];
	
	int req_turn[32];
	int rep_turn[32];
	#endif
};

class mem_fetch;

class partition_mf_allocator : public mem_fetch_allocator {
public:
    partition_mf_allocator( const memory_config *config )
    {
        m_memory_config = config;
    }
    virtual mem_fetch * alloc(const class warp_inst_t &inst, const mem_access_t &access) const 
    {
        abort();
        return NULL;
    }
    virtual mem_fetch * alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr) const;
private:
    const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM channel. 
// - It arbitrates the DRAM channel among multiple sub partitions.  
// - It does not connect directly with the interconnection network. 
class memory_partition_unit
{
public: 
   memory_partition_unit( unsigned partition_id, const struct memory_config *config, class memory_stats_t *stats );
   ~memory_partition_unit(); 

   bool busy() const;

   void cache_cycle( unsigned cycle );
   void dram_cycle();

   void set_done( mem_fetch *mf );

   void visualizer_print( gzFile visualizer_file ) const;
   void print_stat( FILE *fp ) const;
  // { m_dram->print_stat(fp); }
//   void visualize() const { m_dram->visualize(); }
   void print( FILE *fp ) const;
   void print_kain()
   {   
   		assert(0);
        //m_dram->print_kain(); 
   }  
   float KAIN_app1_bw_util()
   {
   		assert(0);//Need to be implemented in HBM
        //return m_dram->KAIN_app1_bw_util(); 
   }
   float KAIN_app2_bw_util()
   {
   		assert(0);//Need to be implemented in HBM
        //return m_dram->KAIN_app2_bw_util(); 
   }

   class memory_sub_partition * get_sub_partition(int sub_partition_id) 
   {
      return m_sub_partition[sub_partition_id]; 
   }

   // Power model
   void set_dram_power_stats(unsigned &n_cmd,
                             unsigned &n_activity,
                             unsigned &n_nop,
                             unsigned &n_act,
                             unsigned &n_pre,
                             unsigned &n_rd,
                             unsigned &n_wr,
                             unsigned &n_req) const;

   int global_sub_partition_id_to_local_id(int global_sub_partition_id) const; 

   unsigned get_mpid() const { return m_id; }

private: 

   unsigned m_id;
   const struct memory_config *m_config;
   class memory_stats_t *m_stats;
   class memory_sub_partition **m_sub_partition; 
//   class dram_t *m_dram;
	class GpuWrapper *m_dram_r;
    //struct kain_cache_block{
    //    new_addr_type m_tag;
    //};
 //   new_addr_type *kain_cache;
   class arbitration_metadata
   {
   public: 
      arbitration_metadata(const struct memory_config *config); 

      // check if a subpartition still has credit 
      bool has_credits(int inner_sub_partition_id) const; 
      // borrow a credit for a subpartition 
      void borrow_credit(int inner_sub_partition_id); 
      // return a credit from a subpartition 
      void return_credit(int inner_sub_partition_id); 

      // return the last subpartition that borrowed credit 
      int last_borrower() const { return m_last_borrower; } 

      void print( FILE *fp ) const; 
   private: 
      // id of the last subpartition that borrowed credit 
      int m_last_borrower; 

      int m_shared_credit_limit; 
      int m_private_credit_limit; 

      // credits borrowed by the subpartitions
      std::vector<int> m_private_credit; 
      int m_shared_credit; 
   }; 
   arbitration_metadata m_arbitration_metadata; 

   // determine wheither a given subpartition can issue to DRAM 
   bool can_issue_to_dram(int inner_sub_partition_id); 

   // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
   struct dram_delay_t
   {
      unsigned long long ready_cycle;
      class mem_fetch* req;
   };
   std::list<dram_delay_t> m_dram_latency_queue;
//   std::list<mem_fetch*> KAIN_HBM_Cache_request;
};

class memory_sub_partition
{
public:
   memory_sub_partition( unsigned sub_partition_id, const struct memory_config *config, class memory_stats_t *stats );
   ~memory_sub_partition(); 

   unsigned get_id() const { return m_id; } 

   bool busy() const;

   void cache_cycle( unsigned cycle );

   bool full() const;
   void push( class mem_fetch* mf, unsigned long long clock_cycle );
   class mem_fetch* pop(); 
   class mem_fetch* top();
   void set_done( mem_fetch *mf );

   unsigned flushL2();

   // interface to L2_dram_queue
   bool L2_dram_queue_empty() const; 
   class mem_fetch* L2_dram_queue_top() const; 
   void L2_dram_queue_pop(); 

   // interface to dram_L2_queue
   bool dram_L2_queue_full() const; 
   void dram_L2_queue_push( class mem_fetch* mf ); 

   void visualizer_print( gzFile visualizer_file );
   void print_cache_stat(unsigned &accesses, unsigned &misses) const;
   void print( FILE *fp ) const;

   void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
   void get_L2cache_sub_stats(struct cache_sub_stats &css) const;
   void get_L2cache_sub_stats_kain(unsigned cluster_id, struct cache_sub_stats &css) const;
   void clear_L2cache_sub_stats_kain();

private:
// data
   unsigned m_id;  //< the global sub partition ID
   const struct memory_config *m_config;
   class l2_cache *m_L2cache;
   class L2interface *m_L2interface;
   partition_mf_allocator *m_mf_allocator;

   // model delay of ROP units with a fixed latency
   struct rop_delay_t
   {
    	unsigned long long ready_cycle;
    	class mem_fetch* req;
   };
   std::queue<rop_delay_t> m_rop;

   // these are various FIFOs between units within a memory partition
   fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_dram_queue;
   fifo_pipeline<mem_fetch> *m_dram_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_icnt_queue; // L2 cache hit response queue

   class mem_fetch *L2dramout; 
   unsigned long long int wb_addr;

   class memory_stats_t *m_stats;

   std::set<mem_fetch*> m_request_tracker;

   friend class L2interface;

public:
   void set_mk_scheduler(MKScheduler* mk_sched);
};

class L2interface : public mem_fetch_interface {
public:
    L2interface( memory_sub_partition *unit ) { m_unit=unit; }
    virtual ~L2interface() {}
    virtual bool full( unsigned size, bool write) const 
    {
        // assume read and write packets all same size
        return m_unit->m_L2_dram_queue->full();
    }
    virtual void push(mem_fetch *mf) 
    {
        mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,0/*FIXME*/);
        m_unit->m_L2_dram_queue->push(mf);
	
	    //printf("ZSQ: m_L2_dram_queue->push(mf) in L2 cache cycle(): ");
	    //mf->mf_print();
	    fflush(stdout);
    }
private:
    memory_sub_partition *m_unit;
};

#endif
