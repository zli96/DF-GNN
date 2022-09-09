log=log
code=main.py
logname=relu_index_mul
# dump_file=(profiling_graph_executor_impl tensorexpr_fuser graph_fuser partition)
dump_file=(profiling_graph_executor_impl manager partition)


# graph dump & update & debug
for i in ${dump_file[@]}; do
    rm ${log}/${logname}_${i}_graph_dump.log ${log}/${logname}_${i}_graph_update.log ${log}/${logname}_${i}_graph_debug.log
    PYTORCH_JIT_LOG_LEVEL="${i}" python ${code} > ${log}/${logname}_${i}_graph_dump.log 2>&1
    PYTORCH_JIT_LOG_LEVEL=">${i}" python ${code} > ${log}/${logname}_${i}_graph_update.log 2>&1
    PYTORCH_JIT_LOG_LEVEL=">>${i}" python ${code} > ${log}/${logname}_${i}_graph_debug.log 2>&1
done
# replace the comment
# sed -r -i 's/^\[[^\[]*\] //g' ${log}/${logname}_*
