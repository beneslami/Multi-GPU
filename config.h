#ifndef CONFIG_H
#define CONFIG_H

#define KAIN_chiplet_frequency		6000000000	//Hz; CHIPLET_frequency in gpu-sim.cc
#define INTER_DIE_TOPOLOGY    		1	        //0-mesh, 1-ring; l2cache.h chiplet_cycle_*()

#define DECOUPLE_NEAR_REMOTE		1		//0-Request/Reply[32], 1-Request/Reply_n/r[32]; l2cache.h
#define threshold_turn			1		//0 to threshold_turn-1: near
#define bound_turn			2		//threshold to bound-1: remote, turn = (turn+1)%bound

#define HBM_CACHE			1		//0: no HBM remote data cache; 1: use HBM_Cache_request[32]

#define CHIPLET_NUM			4		//chiplet number, currently fixed to 4
#define REMOTE_CACHE			0		//l1.5 cache
#define REMOTE_CACHE_QUEUE_LENGTH	256		//remote_cache_request/reply[CHIPLET_NUM], length of each queue per chiplet 
#define REMOTE_CACHE_WIDTH		32		//the request number l1.5 can handle per cycle

#endif
