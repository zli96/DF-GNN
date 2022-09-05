# index_mul demo

`main.py`: index_select + matmul demo

`nvfuser_demo.py`: a demo to test JIT debug log

`pattern_match.py`: pattern match + replace to rewrite the compute graph

## Build code

``` bash
mkdir build && cd build
cmake ..
make -j
```

## Run code

``` bash
source run.sh
```

## JIT debug log

Reference: `torch/csrc/jit/codegen/cuda/README.md`

### PYTORCH_JIT_LOG_LEVEL
* Reference: `torch/csrc/jit/jit_log.cpp, torch/csrc/jit/jit_log.h`
* JIT 中定义了三种log level，graph dump & update & debug，可以查看JIT在不同pass时的执行情况
* Graph dump
    * `PYTORCH_JIT_LOG_LEVEL="profiling_graph_executor_impl" python your_script.py > tmp.log 2>&1`
    * 查看profile graph/optimized graph(至少跑两次)
* Graph update
    * `PYTORCH_JIT_LOG_LEVEL=">profiling_graph_executor_impl" python your_script.py > tmp.log 2>&1`
* Graph debug
    * `PYTORCH_JIT_LOG_LEVEL=">>profiling_graph_executor_impl" python your_script.py > tmp.log 2>&1`
* `PYTORCH_JIT_LOG_LEVEL="graph_fuser" python your_script.py &> tmp.log`
    * Looks for graph dumped with `Before Fusion` & `Before Compilation`, which shows the portion of graph where fusion pass runs on and the result of fusion (`CudaFusionGroup`).

### 查看fuse后的cuda代码
* Ref: [pytorch_source_path]/torch/csrc/jit/codegen/cuda/utils.cpp
* export  PYTORCH_NVFUSER_DUMP="cuda_kernel"

### Disable the nvfuser fusion
* There are three ways to disable nvfuser. Listed below with descending priorities
    * Force using NNC instead of nvfuser for GPU fusion with env variable `export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1`.
    * Disabling nvfuser with torch API `torch._C._jit_set_nvfuser_enabled(False)`.
    * Disable nvfuser with env variable `export PYTORCH_JIT_ENABLE_NVFUSER=0`.