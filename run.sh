log=log
code=main.py

logname=add_relu
dump_file=profiling_graph_executor_impl:tensorexpr_fuser
# # graph debug
# PYTORCH_JIT_LOG_LEVEL=">>tensorexpr_fuser" python ${code} >& ${log}/${logname}_graph_debug.log

# PYTORCH_JIT_LOG_LEVEL=">tensorexpr_fuser" python ${code} >& ${log}/${logname}_graph_update.log

# PYTORCH_JIT_LOG_LEVEL="tensorexpr_fuser" python ${code} >& ${log}/${logname}_graph_dump.log

# graph dump & update & debug
PYTORCH_JIT_LOG_LEVEL="${dump_file}" python ${code} >& ${log}/${logname}_graph_dump.log
PYTORCH_JIT_LOG_LEVEL=">${dump_file}" python ${code} >& ${log}/${logname}_graph_update.log
PYTORCH_JIT_LOG_LEVEL=">>${dump_file}" python ${code} >& ${log}/${logname}_graph_debug.log



