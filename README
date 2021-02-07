0. Before start
Modify $GPGPUSIM_ROOT at setup_environment file.
Make sure $CUDA_INSTALL_PATH is set to correct path

1. To BUILD
make; make;
Because dependency check is not complete for header files, make clean; has to run if header file is modified.

2. To RUN
source setup_environment
./multikernel-sim -l launcher.config -g gpgpusim.config

Note that executable is generated under bin/ directory, and
example config files are in config/ directory.
setup_environment will add lib/ directory in the $LD_LIBRARY_PATH, which contains libcudart.so

3. To modify
3.1. Directory structure
src/ : original gpgpu-sim
launcher/ : classes for multikernel-sim
  |- mk-sched/ : classes for multikernel scheduling. Fixed, Even, Smart are implemented already. Use these as reference implementation for new scheduler
common/ : data structures commonly used by original gpgpu-sim, and launcher
