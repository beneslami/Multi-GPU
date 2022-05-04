#include <map>

#include "gpu_wrapper.h"
#include "Config.h"
#include "Request.h"
#include "MemoryFactory.h"
#include "Memory.h"
#include "DDR3.h"
#include "DDR4.h"
#include "LPDDR3.h"
#include "LPDDR4.h"
#include "GDDR5.h"
#include "WideIO.h"
#include "WideIO2.h"
#include "HBM.h"
#include "SALP.h"


// using namespace ramulator;

extern unsigned long long returnq_in;

static map<string, function<MemoryBase *(const Config&, int, fifo_pipeline<mem_fetch> *)> > name_to_func = {
    {"DDR3", &MemoryFactory<DDR3>::create}, {"DDR4", &MemoryFactory<DDR4>::create},
    {"LPDDR3", &MemoryFactory<LPDDR3>::create}, {"LPDDR4", &MemoryFactory<LPDDR4>::create},
    {"GDDR5", &MemoryFactory<GDDR5>::create},
    {"WideIO", &MemoryFactory<WideIO>::create}, {"WideIO2", &MemoryFactory<WideIO2>::create},
    {"HBM", &MemoryFactory<HBM>::create},
    {"SALP-1", &MemoryFactory<SALP>::create}, {"SALP-2", &MemoryFactory<SALP>::create}, {"SALP-MASA", &MemoryFactory<SALP>::create},
};


GpuWrapper::GpuWrapper(const Config& configs, int cacheline,  memory_partition_unit *mp, unsigned id):
    read_cb_func(std::bind(&GpuWrapper::readComplete, this, std::placeholders::_1)),
    write_cb_func(std::bind(&GpuWrapper::writeComplete, this, std::placeholders::_1))
{
    const string& std_name = configs["standard"];
    assert(name_to_func.find(std_name) != name_to_func.end() && "unrecognized standard name");

//    read_cb_func(std::bind(&GpuWrapper::readComplete, this, std::placeholders::_1));
//    write_cb_func(std::bind(&GpuWrapper::writeComplete, this, std::placeholders::_1));
    r_returnq = new fifo_pipeline<mem_fetch>("ramulatorreturnq", 0, 4096);
    mem = name_to_func[std_name](configs, cacheline, r_returnq);
    tCK = mem->clk_ns();
    m_memory_partition_unit = mp;
    mem_id = id;
}


GpuWrapper::~GpuWrapper() {
    delete mem;
}

void GpuWrapper::cycle()
{
    mem->tick();
    Stats_Ram::curTick++;
}

bool GpuWrapper::send(Request req)
{
    return mem->send(req);
}

void GpuWrapper::finish(void) {
    fprintf(stdout, "The dram stat is herere!!!!!!!!!!\n" );
    mem->finish();
    Stats_Ram::statlist.printall();
}

bool GpuWrapper::full(int request_type, long request_addr )
{
    // 1 is for write, while 0 for read
    if (request_type == 0)
    {
        return mem->full(Request::Type::READ, request_addr);
    } else {
        return mem->full(Request::Type::WRITE, request_addr);
    }
}

// Request GpuWrapper::address_convert(std::mem_fetch &mf) {
//     if (mf.is_write())
//     {
//         Request req((long)mf.kain_get_addr(), Request::Type::WRITE, write_cb_func, mf.get_sid());
//         return req;
//     } else {
//         Request req((long)mf.kain_get_addr(), Request::Type::READ, read_cb_func, mf.get_sid());
//         return req;
//     }
// }

//In this function, I can put the request into the DRAM to L2 queue.
void GpuWrapper::readComplete(Request& req) {
    //if(!r_returnq->full()) FIX ME!!!!!!!!!!
    // fprintf(stderr,"enter here_READ COMPLETE`\n");
    //fprintf(stderr, "Readthe returned req addres is %ld\n", req.mf->kain_get_addr());
//    fprintf(stdout, "KAIN read complete begin\n");
//    fflush(stdout);

    auto& mf_queue = mem_temp_r.find(req.mf->kain_get_addr())->second;
    //mem_fetch* mf = mf_queue.front();
    mem_fetch* mf = req.mf;

    mf_queue.pop_front();
    if (!mf_queue.size())
        mem_temp_r.erase(req.mf->kain_get_addr()) ;
    assert(mf!=NULL);
    mf->set_status(IN_PARTITION_MC_RETURNQ, gpu_sim_cycle + gpu_tot_sim_cycle);
    mf->set_reply();
    r_returnq->push(mf);
    returnq_in++;
    //dram_L2_queue_push(mf);
}

void GpuWrapper::writeComplete(Request& req) {
    //fprintf(stderr,"enter here_Write COMPLETE`\n");
    //fprintf(stderr, "Writethe returned req addres is %ld\n", req.mf->kain_get_addr());
//    fprintf(stdout, "KAIN write complete begin\n");
//    fflush(stdout);
    auto& mf_queue = mem_temp_w.find(req.mf->kain_get_addr())->second;
    //mem_fetch* mf = mf_queue.front();
    mem_fetch* mf = req.mf;
    mf_queue.pop_front();
    if (!mf_queue.size())
        mem_temp_w.erase(req.mf->kain_get_addr()) ;
    mf->set_status(IN_PARTITION_MC_RETURNQ, gpu_sim_cycle + gpu_tot_sim_cycle);
#if SM_SIDE_LLC == 0
    if (!( mf->get_access_type() != L1_WRBK_ACC && mf->get_access_type() != L2_WRBK_ACC)) {
        m_memory_partition_unit->set_done(mf);
        delete mf;
    } else {
        mf->set_reply();
        r_returnq->push(mf);
        returnq_in++;
    }
#endif

#if SM_SIDE_LLC == 1
    if (mf->get_sid()/32 == mf->get_chip_id()/8) {
	if (!( mf->get_access_type() != L1_WRBK_ACC && mf->get_access_type() != L2_WRBK_ACC)) {
	    m_memory_partition_unit->set_done(mf);
	    delete mf;
	} else {
	    mf->set_reply();
	    r_returnq->push(mf);
    	    returnq_in++;
	}
    } else {
        mf->set_reply();
        r_returnq->push(mf);
        returnq_in++;
    }
#endif

//    fprintf(stdout, "KAIN write complete end\n");
//    fflush(stdout);
}


void GpuWrapper::push(mem_fetch* mf)
{
    //fprintf(stderr, "mem id is %u\n", this->mem_id);
    Request *req;
//    fprintf(stdout, "The core number is %d, get_sid %d, is write %d\n", core_numbers, mf->get_sid(), mf->is_write());
//    fflush(stdout);

    if (mf->is_write())
    {
        if ((mf->get_sid()) > (unsigned)core_numbers)
        {
            req = new Request((long)mf->kain_get_addr(), Request::Type::WRITE, write_cb_func, core_numbers);
        }
        else {

            req = new Request((long)mf->kain_get_addr(), Request::Type::WRITE, write_cb_func, mf->get_sid());
        }
    } else {

        if ((mf->get_sid())  > (unsigned)core_numbers)
        {
            //req = new Request((long)mf->kain_get_addr(), Request::Type::WRITE, write_cb_func, core_numbers);
            req = new Request((long)mf->kain_get_addr(), Request::Type::READ, read_cb_func, core_numbers);
        }
        else {
            req = new Request((long)mf->kain_get_addr(), Request::Type::READ, read_cb_func, mf->get_sid());
        }
    }
    assert(req->coreid >= 0);
    req->mf = mf;
    bool accepted = send(*req);
    assert(accepted);
    if (accepted)
    {
        if (mf->is_write()) {
            mem_temp_w[mf->kain_get_addr()].push_back(mf);
            // fprintf(stderr, "the pushed_write req addres is %ld \n", mf->kain_get_addr());
        } else {
            mem_temp_r[mf->kain_get_addr()].push_back(mf);
            //fprintf(stderr, "the pushed_read req addres is %ld\n", mf->kain_get_addr());
        }
    }

    delete req;
}

bool GpuWrapper::r_returnq_full() const
{
    if(r_returnq->get_length() < r_returnq->get_max_len()-1024)
        return false;
    else
        return true;
//    return r_returnq->full();
}

int GpuWrapper::r_returnq_size() const 
{
    return r_returnq->get_length();
}

class mem_fetch* GpuWrapper::r_return_queue_top() const
{
    return r_returnq->top();
}

class mem_fetch* GpuWrapper::r_return_queue_pop() const
{
    return r_returnq->pop();
}
