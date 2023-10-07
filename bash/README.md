# 测试脚本

运行时间性能测试脚本
* run_nolog.sh：无log运行
* run_multi.sh：遍历同一数据集下的bs和实现方案
* run_subgraph.sh：测试subgraph method的性能
* run_full_graph.sh：测试在非batch graph上的性能

图性质统计脚本
* run_graph_statics.sh：统计不同数据集的图性质（节点数、邻居数、图密度）

Kernel性能分析脚本
* run_ncu.sh：运行nsight compute的分析
* run_nsys.sh：运行nsight system的分析

