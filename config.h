#ifndef CONFIG_H
#define CONFIG_H

#define KAIN_chiplet_frequency		6000000000	//Hz; CHIPLET_frequency in gpu-sim.cc
#define STATISTICS  0
#define INTER_DIE_TOPOLOGY    		1	        //0-mesh, 1-ring; l2cache.h chiplet_cycle_*()
#define DECOUPLE_NEAR_REMOTE		0		//0-Request/Reply[32], 1-Request/Reply_n/r[32]; l2cache.h
#define threshold_turn			1		//0 to threshold_turn-1: near
#define bound_turn			2		//threshold to bound-1: remote, turn = (turn+1)%bound
#define HBM_CACHE			0		//0: no HBM remote data cache; 1: use HBM_Cache_request[32]

#define CHIPLET_NUM			4		//chiplet number, currently fixed to 4

#define SUB_ID_DEC 			0		//0 for mem-side-shared, 1 for sm-side-shared, 2 for mem-side_private, 3 for sm-side-private
#define SM_SIDE_LLC 			0		//0 for mem-side, 1 for sm-side
#define ICNT_FREQ_CTRL			8		//24576MHz/1024MHz
//#define ICNT_FREQ_CTRL			3		//3072MHz/1024MHz
//#define ICNT_FREQ_CTRL			6		//6144MHz/1024MHz
//#define ICNT_FREQ_CTRL			4		//4096MHz/1024MHz
#define INTER_TOPO			1		//0 for full connection, 1 for ring
#define INTER_DELAY			32		//inter link delay (default 32 cycles per hop)

#define REMOTE_CACHE			0		//l1.5 cache, only applied in mem-side-llc
#define REMOTE_CACHE_ENTRY		65536		//32MB total L1.5; 65536(cacheline per module) * 128B (cacheline size) * 4 (module number)
//#define REMOTE_CACHE_ENTRY		16384		//8MB total L1.5; 16384(cacheline per module) * 128B (cacheline size) * 4 (module number)
#define RC_BUS_WIDTH			8		//remote_cache[i] request/reply pop number per core cycle

#define PAGE_ALLOC			0		//0 for round robin, 1 for first touch
#define CTA_ALLOC			1		//0 for module round robin, 1 for centralized distribution
#define BEN_OUTPUT          1
#endif
