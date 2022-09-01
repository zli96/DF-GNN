log=log
code=main.py
logname=linear_add_relu
dump_file=(profiling_graph_executor_impl tensorexpr_fuser)

# graph dump & update & debug
rm ${log}/${logname}_graph_dump.log ${log}/${logname}_graph_update.log ${log}/${logname}_graph_debug.log
for i in ${dump_file[@]}; do
    PYTORCH_JIT_LOG_LEVEL="${i}" python ${code} >> ${log}/${logname}_graph_dump.log 2>&1
    PYTORCH_JIT_LOG_LEVEL=">${i}" python ${code} >> ${log}/${logname}_graph_update.log 2>&1
    PYTORCH_JIT_LOG_LEVEL=">>${i}" python ${code} >> ${log}/${logname}_graph_debug.log 2>&1
done




