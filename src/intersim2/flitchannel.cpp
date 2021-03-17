// $Id: flitchannel.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// ----------------------------------------------------------------------
//
//  File Name: flitchannel.cpp
//  Author: James Balfour, Rebecca Schultz
//
// ----------------------------------------------------------------------

#include "flitchannel.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include "router.hpp"
#include "globals.hpp"
#include "flit.hpp"
#include "gpuicnt.h"
#include "../gpgpu-sim/mem_fetch.h"
extern unsigned long long gpu_sim_cycle;
InterGPU *igpu10 = new InterGPU();

// ----------------------------------------------------------------------
//  $Author: jbalfour $
//  $Date: 2007/06/27 23:10:17 $
//  $Id: flitchannel.cpp 5188 2012-08-30 00:31:31Z dub $
// ----------------------------------------------------------------------
FlitChannel::FlitChannel(Module * parent, string const & name, int classes)
        : Channel<Flit>(parent, name), _routerSource(NULL), _routerSourcePort(-1),
          _routerSink(NULL), _routerSinkPort(-1), _idle(0), _classes(classes) {
    _active.resize(classes, 0);
}

void FlitChannel::SetSource(Router const * const router, int port) {
    _routerSource = router;
    _routerSourcePort = port;
}

void FlitChannel::SetSink(Router const * const router, int port) {
    _routerSink = router;
    _routerSinkPort = port;
}

void FlitChannel::Send(Flit * f) {
    if (f) {
        ++_active[f->cl];
    } else {
        ++_idle;
    }
    if(f->head){
        mem_fetch *temp = static_cast<mem_fetch *>(f->data);
        if(temp->is_remote()) {
            std::ostringstream out;
            std::cout << "_input_write\tsrc: " << f->src << "\tdst: " << f->dest << "\tpacket_ID: "
                      << temp->get_request_uid() << "cycle: " << gpu_sim_cycle << "\n";
            out << "input_write\tsrc: " << f->src << "\tdst: " << f->dest << "\tpacket_ID: "
                << temp->get_request_uid() << "\ttype: "<< temp->get_type() <<"\tcycle: " << gpu_sim_cycle << "\n";
            igpu10->apply(out.str().c_str());
        }
    }
    Channel<Flit>::Send(f);
}

void FlitChannel::ReadInputs() {
    Flit const *const &f = _input;
    if (f && f->watch) {
        *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                   << "Beginning channel traversal for flit " << f->id
                   << " with delay " << _delay
                   << "." << endl;

        if(f->head){
            mem_fetch *temp = static_cast<mem_fetch *>(f->data);
            if(temp->is_remote()) {
                std::ostringstream out;
                std::cout << "waiting_buffer_push\tsrc: " << f->src << "\tdst: " << f->dest << "\tpacket_ID: "
                     << temp->get_request_uid() << "cycle: " << gpu_sim_cycle << "\n";
                out << "waiting_buffer_push\tsrc: " << f->src << "\tdst: " << f->dest << "\tpacket_ID: "
                    << temp->get_request_uid() << "\ttype: "<< temp->get_type() <<"\tcycle: " << gpu_sim_cycle << "\n";
                igpu10->apply(out.str().c_str());
            }
        }
    }
    Channel<Flit>::ReadInputs();
}

void FlitChannel::WriteOutputs() {
    Channel<Flit>::WriteOutputs();
    if (_output && _output->watch) {
        *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                   << "Completed channel traversal for flit " << _output->id
                   << "." << endl;
        Flit const *const &f = _output;
        if(f->head){
            mem_fetch *temp = static_cast<mem_fetch *>(f->data);
            if(temp->is_remote()) {
                std::ostringstream out;
                std::cout << "waiting_buffer_pop\tsrc: " << f->src << "\tdst: " << f->dest << "\tpacket_ID: "
                     << temp->get_request_uid() << "cycle: " << gpu_sim_cycle << "\n";
                out << "waiting_buffer_pop\tsrc: " << f->src << "\tdst: " << f->dest << "\tpacket_ID: "
                    << temp->get_request_uid() << "\ttype: "<< temp->get_type() <<"\tcycle: " << gpu_sim_cycle << "\n";
                igpu10->apply(out.str().c_str());
            }
        }
    }
}
