// Copyright (c) 2009-2013, Tor M. Aamodt, Dongdong Li, Ali Bakhoda
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

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "interconnect_interface.hpp"
#include "routefunc.hpp"
#include "globals.hpp"
#include "trafficmanager.hpp"
#include "power_module.hpp"
#include "mem_fetch.h"
#include "flit.hpp"
#include "gputrafficmanager.hpp"
#include "booksim.hpp"
#include "intersim_config.hpp"
#include "network.hpp"
#include "../../config.h"
extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;

InterconnectInterface* InterconnectInterface::New(const char* const config_file)
{
  if (! config_file ) {
    cout << "Interconnect Requires a configfile" << endl;
    exit (-1);
  }
  InterconnectInterface* icnt_interface = new InterconnectInterface();
  icnt_interface->_icnt_config = new IntersimConfig();
  icnt_interface->_icnt_config->ParseFile(config_file);
  
  return icnt_interface;
}

InterconnectInterface::InterconnectInterface()
{
  
}

InterconnectInterface::~InterconnectInterface()
{
  for (int i=0; i<_subnets; ++i) {
    ///Power analysis
    if(_icnt_config->GetInt("sim_power") > 0){
      Power_Module pnet(_net[i], *_icnt_config);
      pnet.run();
    }
    delete _net[i];
  }
  
  delete _traffic_manager;
  _traffic_manager = NULL;
  delete _icnt_config;
}

void InterconnectInterface::CreateInterconnect(unsigned n_shader, unsigned n_mem)
{
  _n_shader = n_shader;
  _n_mem = n_mem;
  
  InitializeRoutingMap(*_icnt_config);
  
  gPrintActivity = (_icnt_config->GetInt("print_activity") > 0);
  gTrace = (_icnt_config->GetInt("viewer_trace") > 0);
  
  string watch_out_file = _icnt_config->GetStr( "watch_out" );
  if(watch_out_file == "") {
    gWatchOut = NULL;
  } else if(watch_out_file == "-") {
    gWatchOut = &cout;
  } else {
    gWatchOut = new ofstream(watch_out_file.c_str());
  }
  
  _subnets = _icnt_config->GetInt("subnets");
  assert(_subnets);
  
  /*To include a new network, must register the network here
   *add an else if statement with the name of the network
   */
  _net.resize(_subnets);
  for (int i = 0; i < _subnets; ++i) {
    ostringstream name;
    name << "network_" << i;
    _net[i] = Network::New( *_icnt_config, name.str() );
  }
  
  assert(_icnt_config->GetStr("sim_type") == "gpgpusim");
  _traffic_manager = static_cast<GPUTrafficManager*>(TrafficManager::New( *_icnt_config, _net )) ;
  
  _flit_size = _icnt_config->GetInt( "flit_size" );
  
  // Config for interface buffers
  if (_icnt_config->GetInt("ejection_buffer_size")) {
    _ejection_buffer_capacity = _icnt_config->GetInt( "ejection_buffer_size" ) ;
  } else {
    _ejection_buffer_capacity = _icnt_config->GetInt( "vc_buf_size" );
  }
  
  _boundary_buffer_capacity = _icnt_config->GetInt( "boundary_buffer_size" ) ;
  assert(_boundary_buffer_capacity);
  if (_icnt_config->GetInt("input_buffer_size")) {
    _input_buffer_capacity = _icnt_config->GetInt("input_buffer_size");
  } else {
    _input_buffer_capacity = 9;
  }
  _vcs = _icnt_config->GetInt("num_vcs");
  
  _CreateBuffer();
  _CreateNodeMap(_n_shader, _n_mem, _traffic_manager->_nodes, _icnt_config->GetInt("use_map"));
}

void InterconnectInterface::Init()
{
  _traffic_manager->Init();
  // TODO: Should we init _round_robin_turn?
  //       _boundary_buffer, _ejection_buffer and _ejected_flit_queue should be cleared
}

void InterconnectInterface::Push(unsigned input_deviceID, unsigned output_deviceID, void *data, unsigned int size)
{
  // it should have free buffer
  assert(HasBuffer(input_deviceID, size));
  int output_icntID = _node_map[output_deviceID];
  int input_icntID = _node_map[input_deviceID];
  
#if 0
  cout<<"Call interconnect push input: "<<input<<" output: "<<output<<endl;
#endif
  
  //TODO: move to _IssuePacket
  //TODO: create a Inject and wrap _IssuePacket and _GeneratePacket
  unsigned int n_flits = size / _flit_size + ((size % _flit_size)? 1:0);
  //if(n_flits != 1)//KAIN
  //{
  //  printf("size is %d\n",size); 
  //  assert(0);
  //}

  int subnet;
  if (_subnets == 1) {
    subnet = 0;
  } else {
    if (input_deviceID < _n_shader) {
      subnet = 0;
    } else if (_n_shader+_n_mem <= input_deviceID && input_deviceID < _n_shader+_n_mem+4) {
      subnet = 0;
    } else {
      subnet = 1;
    }
  }
  //TODO: Remove mem_fetch to reduce dependency
  Flit::FlitType packet_type;
  mem_fetch* mf = static_cast<mem_fetch*>(data);
  
  switch (mf->get_type()) {
    case READ_REQUEST:  packet_type = Flit::READ_REQUEST   ;break;
    case WRITE_REQUEST: packet_type = Flit::WRITE_REQUEST  ;break;
    case READ_REPLY:    packet_type = Flit::READ_REPLY     ;break;
    case WRITE_ACK:     packet_type = Flit::WRITE_REPLY    ;break;
    default: assert (0);
  }
  //TODO: _include_queuing ?
  _traffic_manager->_GeneratePacket( input_icntID, -1, 0 /*class*/, _traffic_manager->_time, subnet, n_flits, packet_type, data, output_icntID);
  
		//printf("ZSQ: cycle %llu, Push(%d, %d) subnet %d size = %u, mf sid = %d chip_id = %d sub_partition_id=%u type = %s inst @ pc=0x%04x\n", gpu_sim_cycle+gpu_tot_sim_cycle, input_deviceID, output_deviceID, subnet, size, mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->is_write()?"W":"R", mf->get_pc()); 
		fflush(stdout);

#if DOUB
  cout <<"Traffic[" << subnet << "] (mapped) sending form "<< input_icntID << " to " << output_icntID << endl;
#endif
//  }
}

void* InterconnectInterface::Pop(unsigned deviceID)
{
  int icntID = _node_map[deviceID];
#if DEBUG
  cout<<"Call interconnect POP  " << output<<endl;
#endif
  
  void* data = NULL;
 
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  if (deviceID < _n_shader)
    subnet = 1;
 
  int turn = _round_robin_turn[subnet][icntID];
  for (int vc=0;(vc<_vcs) && (data==NULL);vc++) {
    if (_boundary_buffer[subnet][icntID][turn].HasPacket()) {
      data = _boundary_buffer[subnet][icntID][turn].PopPacket();
    }
    turn++;
    if (turn == _vcs) turn = 0;
  }
  if (data) {
    _round_robin_turn[subnet][icntID] = turn;
  }

  if(data) {
    mem_fetch* mf = static_cast<mem_fetch*>(data);
    //printf("ZSQ: cycle %llu, Pop(%d), subnet %d, mf sid = %d chip_id = %d sub_partition_id=%u type = %s inst @ pc=0x%04x\n", gpu_sim_cycle+gpu_tot_sim_cycle, deviceID, subnet, mf->get_sid(), mf->get_chip_id(), mf->get_sub_partition_id(), mf->is_write()?"W":"R", mf->get_pc());
    fflush(stdout);
  } 

  return data;
}

void InterconnectInterface::Advance()
{
  _traffic_manager->_Step();
}

bool InterconnectInterface::Busy() const
{
  bool busy = !_traffic_manager->_total_in_flight_flits[0].empty();
  if (!busy) {
    for (int s = 0; s < _subnets; ++s) {
      for (unsigned n = 0; n < _n_shader+_n_mem; ++n) {
        //FIXME: if this cannot make sure _partial_packets is empty
        assert(_traffic_manager->_input_queue[s][n][0].empty());
      }
    }
  } else {
    return true;
  }
  for (int s = 0; s < _subnets; ++s) {
    for (unsigned n=0; n < (_n_shader+_n_mem+4); ++n) {
      for (int vc=0; vc<_vcs; ++vc) {
        if (_boundary_buffer[s][n][vc].HasPacket() ) {
          return true;
        }
      }
    }
  }
  return false;
}

bool InterconnectInterface::HasBuffer(unsigned deviceID, unsigned int size) const
{
  bool has_buffer = false;
  unsigned int n_flits = size / _flit_size + ((size % _flit_size)? 1:0);
  int icntID = _node_map.find(deviceID)->second;
  has_buffer = _traffic_manager->_input_queue[0][icntID][0].size() +n_flits <= _input_buffer_capacity;
  
  if ((_subnets>1) && deviceID >= _n_shader && deviceID < _n_shader+_n_mem) // deviceID is memory node
    has_buffer = _traffic_manager->_input_queue[1][icntID][0].size() +n_flits <= _input_buffer_capacity;
/*
  if ((_subnets>1) && deviceID >= _n_shader && deviceID < _n_shader+_n_mem)
    printf("ZSQ: HasBuffer(%u, size %u) subnet 1, input_queue.size %u, n_flits %u, input_buffer_capacity %u\n", deviceID, size, _traffic_manager->_input_queue[1][icntID][0].size(), n_flits, _input_buffer_capacity);
  else
    printf("ZSQ: HasBuffer(%u, size %u) subnet 0, input_queue.size %u, n_flits %u, input_buffer_capacity %u\n", deviceID, size, _traffic_manager->_input_queue[0][icntID][0].size(), n_flits, _input_buffer_capacity);
  fflush(stdout);
*/  
  return has_buffer;
}

void InterconnectInterface::DisplayStats() const
{
  _traffic_manager->UpdateStats();
  _traffic_manager->DisplayStats();
}

unsigned InterconnectInterface::GetFlitSize() const
{
  return _flit_size;
}

void InterconnectInterface::DisplayOverallStats() const
{
  // hack: booksim2 use _drain_time and calculate delta time based on it, but we don't, change this if you have a better idea
  _traffic_manager->_drain_time = _traffic_manager->_time;
  // hack: also _total_sims equals to number of kernel calls
  _traffic_manager->_total_sims += 1;
  
  _traffic_manager->_UpdateOverallStats();
  _traffic_manager->DisplayOverallStats();
  if(_traffic_manager->_print_csv_results) {
    _traffic_manager->DisplayOverallStatsCSV();
  }
}

void InterconnectInterface::DisplayState(FILE *fp) const
{
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n\n");
  fprintf(fp, "In-flight Flits = %lu\n", _traffic_manager->_total_in_flight_flits[0].size());
  for (map<int, Flit*>::iterator it = _traffic_manager->_total_in_flight_flits[0].begin(), it_end = _traffic_manager->_total_in_flight_flits[0].end();
       it != it_end; ++it) {
    mem_fetch* mf = static_cast<mem_fetch*>(it->second->data);
    fprintf(fp, "From Shader[%d] Warp[%d]\n", mf->get_sid(), mf->get_wid());
  }

  for (int s = 0; s < _subnets; ++s) {
    for (unsigned n = 0; n < _n_shader+_n_mem; ++n) {
      if (!_traffic_manager->_input_queue[s][n][0].empty()) {
        fprintf(fp, "Input Queue for subnet[%d], N[%d], [0] = %lu\n", s, n, _traffic_manager->_input_queue[s][n][0].size());
      }
    }
  }

  for (int s = 0; s < _subnets; ++s) {
    for (unsigned n=0; n < (_n_shader+_n_mem); ++n) {
      for (int vc=0; vc<_vcs; ++vc) {
        if (_boundary_buffer[s][n][vc].HasPacket() ) {
          fprintf(fp, "Boundary buffer for subnet[%d], N[%d], VC[%d] has packet\n", s, n, vc);
        }
      }
    }
  }
//  fprintf(fp,"GPGPU-Sim uArch: interconnect busy state\n");
  
//  for (unsigned i=0; i<net_c;i++) {
//    if (traffic[i]->_measured_in_flight)
//      fprintf(fp,"   Network %u has %u _measured_in_flight\n", i, traffic[i]->_measured_in_flight );
//  }
//  
//  for (unsigned i=0 ;i<(_n_shader+_n_mem);i++ ) {
//    if( !traffic[0]->_partial_packets[i] [0].empty() )
//      fprintf(fp,"   Network 0 has nonempty _partial_packets[%u][0]\n", i);
//    if ( doub_net && !traffic[1]->_partial_packets[i] [0].empty() )
//      fprintf(fp,"   Network 1 has nonempty _partial_packets[%u][0]\n", i);
//    for (unsigned j=0;j<g_num_vcs;j++ ) {
//      if( !ejection_buf[i][j].empty() )
//        fprintf(fp,"   ejection_buf[%u][%u] is non-empty\n", i, j);
//      if( clock_boundary_buf[i][j].has_packet() )
//        fprintf(fp,"   clock_boundary_buf[%u][%u] has packet\n", i, j );
//    }
//  }
}

void InterconnectInterface::Transfer2BoundaryBuffer(int subnet, int output)
{
  Flit* flit;
  int vc;
  for (vc=0; vc<_vcs;vc++) {
    
    if ( !_ejection_buffer[subnet][output][vc].empty() && _boundary_buffer[subnet][output][vc].Size() < _boundary_buffer_capacity ) {
      flit = _ejection_buffer[subnet][output][vc].front();
      assert(flit);
      
      _ejection_buffer[subnet][output][vc].pop();
      _boundary_buffer[subnet][output][vc].PushFlitData( flit->data, flit->tail);
      
      _ejected_flit_queue[subnet][output].push(flit); //indicate this flit is already popped from ejection buffer and ready for credit return
      
      if ( flit->head ) {
        assert (flit->dest == output);
      }
    }
  }
}

void InterconnectInterface::WriteOutBuffer(int subnet, int output_icntID, Flit*  flit )
{
  int vc = flit->vc;
  assert (_ejection_buffer[subnet][output_icntID][vc].size() < _ejection_buffer_capacity);
  _ejection_buffer[subnet][output_icntID][vc].push(flit);
}

int InterconnectInterface::GetIcntTime() const
{
  return _traffic_manager->getTime();
}

Stats* InterconnectInterface::GetIcntStats(const string &name) const
{
  return _traffic_manager->getStats(name);
}

Flit* InterconnectInterface::GetEjectedFlit(int subnet, int node)
{
  Flit* flit = NULL;
  if (!_ejected_flit_queue[subnet][node].empty()) {
    flit = _ejected_flit_queue[subnet][node].front();
    _ejected_flit_queue[subnet][node].pop();
  }
  return flit;
}

void InterconnectInterface::_CreateBuffer()
{
  unsigned nodes = _n_shader + _n_mem + 4;
  //printf("ZSQ: InterconnectInterface::_CreateBuffer(), nodes = %d + %d + 4 = %d\n", _n_shader, _n_mem, _n_shader + _n_mem + 4);
  fflush(stdout);
  _boundary_buffer.resize(_subnets);
  _ejection_buffer.resize(_subnets);
  _round_robin_turn.resize(_subnets);
  _ejected_flit_queue.resize(_subnets);
  
  for (int subnet = 0; subnet < _subnets; ++subnet) {
    _ejection_buffer[subnet].resize(nodes);
    _boundary_buffer[subnet].resize(nodes);
    _round_robin_turn[subnet].resize(nodes);
    _ejected_flit_queue[subnet].resize(nodes);
    
    for (unsigned node=0;node < nodes;++node){
      _ejection_buffer[subnet][node].resize(_vcs);
      _boundary_buffer[subnet][node].resize(_vcs);
    }
  }
}

void InterconnectInterface::_CreateNodeMap(unsigned n_shader, unsigned n_mem, unsigned n_node, int use_map)
{
  if (use_map) {
    map<unsigned, vector<unsigned> > preset_memory_map;
    
    // good for 8 shaders and 8 memory cores
    {
      unsigned memory_node[] = {1, 3, 4, 6, 9, 11, 12, 14};
      preset_memory_map[16] = vector<unsigned>(memory_node, memory_node+8);
    }
    
    // good for 28 shaders and 8 memory cores
    {
      unsigned memory_node[] = {3, 7, 10, 12, 23, 25, 28, 32};
      preset_memory_map[36] = vector<unsigned>(memory_node, memory_node+8);
    }
    
    // good for 56 shaders and 8 memory cores
    {
      unsigned memory_node[] = {3, 15, 17, 29, 36, 47, 49, 61};
      preset_memory_map[64] = vector<unsigned>(memory_node, memory_node+sizeof(memory_node)/sizeof(unsigned));
    }
    
    // good for 110 shaders and 11 memory cores
    {
      unsigned memory_node[] = {12, 20, 25, 28, 57, 60, 63, 92, 95,100,108};
      preset_memory_map[121] = vector<unsigned>(memory_node, memory_node+sizeof(memory_node)/sizeof(unsigned));
    }

    // good for 80 shaders and 64 memory cores
    {
      unsigned memory_node[] = {0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 46, 48, 50, 52, 54, 56, 58, 62, 64, 66, 68, 70, 72, 76, 78, 80, 82, 84, 86, 88, 90, 92, 96, 98, 100, 102, 104, 106, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130, 132, 134 , 138, 140, 142};
      preset_memory_map[144] = vector<unsigned>(memory_node, memory_node+sizeof(memory_node)/sizeof(unsigned));
    }

    const vector<unsigned> &memory_node = preset_memory_map[n_node];
    if (memory_node.empty()) {
      cerr<<"ERROR!!! NO MAPPING IMPLEMENTED YET FOR THIS CONFIG"<<endl;
      assert(0);
    }
    
    // create node map
    unsigned next_node = 0;
    unsigned memory_node_index = 0;
    for (unsigned i = 0; i < n_shader; ++i) {
      while (next_node == memory_node[memory_node_index]) {
        next_node += 1;
        memory_node_index += 1;
      }
      _node_map[i] = next_node;
      next_node += 1;
    }
    for (unsigned i = n_shader; i < n_shader+n_mem; ++i) {
      _node_map[i] = memory_node[i-n_shader];
    }
  } else { //not use preset map
    for (unsigned i=0;i<n_node;i++) {
      _node_map[i]=i;
    }
  }
  
  for (unsigned i = 0; i < n_node ; i++) {
    for (unsigned j = 0; j< n_node ; j++) {
      if ( _node_map[j] == i ) {
        _reverse_node_map[i]=j;
        break;
      }
    }
  }
  
  //FIXME: should compatible with non-squre number
  _DisplayMap((int) sqrt(n_node), n_node);
  
}

void InterconnectInterface::_DisplayMap(int dim,int count)
{
  cout << "GPGPU-Sim uArch: interconnect node map (shaderID+MemID to icntID)" << endl;
  cout << "GPGPU-Sim uArch: Memory nodes ID start from index: " << _n_shader << endl;
  cout << "GPGPU-Sim uArch: ";
  for (int i = 0;i < count; i++) {
    cout << setw(4) << _node_map[i];
    if ((i+1)%dim == 0 && i != count-1)
      cout << endl << "GPGPU-Sim uArch: ";
  }
  cout << endl;
  
  cout << "GPGPU-Sim uArch: interconnect node reverse map (icntID to shaderID+MemID)" << endl;
  cout << "GPGPU-Sim uArch: Memory nodes start from ID: " << _n_shader << endl;
  cout << "GPGPU-Sim uArch: ";
  for (int i = 0;i < count; i++) {
    cout << setw(4) << _reverse_node_map[i];
    if ((i+1)%dim == 0 && i != count-1)
      cout << endl << "GPGPU-Sim uArch: ";
  }
  cout << endl;
}

void* InterconnectInterface::_BoundaryBufferItem::PopPacket()
{
  assert (_packet_n);
  void * data = NULL;
  void * flit_data = _buffer.front();
  while (data == NULL) {
    assert(flit_data == _buffer.front()); //all flits must belong to the same packet
    if (_tail_flag.front()) {
      data = _buffer.front();
      _packet_n--;
    }
    _buffer.pop();
    _tail_flag.pop();
  }
  return data;
}

void* InterconnectInterface::_BoundaryBufferItem::TopPacket() const
{
  assert (_packet_n);
  void* data = NULL;
  void* temp_d = _buffer.front();
  while (data==NULL) {
    if (_tail_flag.front()) {
      data = _buffer.front();
    }
    assert(temp_d == _buffer.front()); //all flits must belong to the same packet
  }
  return data;
  
}

void InterconnectInterface::_BoundaryBufferItem::PushFlitData(void* data,bool is_tail)
{
  _buffer.push(data);
  _tail_flag.push(is_tail);
  if (is_tail) {
    _packet_n++;
  }
}

int InterconnectInterface::get_icnt_cycle() {
    return _traffic_manager->_time;
}