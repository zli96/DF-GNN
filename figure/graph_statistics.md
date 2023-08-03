
# Graph statistics

- [Graph statistics](#graph-statistics)
  - [molecular dataset](#molecular-dataset)
  - [super-pixel dataset](#super-pixel-dataset)
  - [SBM dataset](#sbm-dataset)
  - [Full-graph dataset](#full-graph-dataset)




## molecular dataset

**ogbg-molhiv**

每个子图平均有25个节点，不同子图间的节点数差距较大

<img decoding="async" src=ogbg-molhiv_num_nodes.png width="60%">

每个节点平均有2个邻居，绝大多数节点的邻居数为2、3、4

<img decoding="async" src=ogbg-molhiv_num_neigh.png width="60%">

## super-pixel dataset

**MNIST**

每个子图平均有70个节点，不同子图间的节点数差距不大

<img decoding="async" src=MNIST_num_nodes.png width="60%">

所有节点的邻居数均为8

<img decoding="async" src=MNIST_num_neigh.png width="60%">

**CIFAR10**

每个子图平均有117个节点，不同子图间的节点数差距不大

<img decoding="async" src=CIFAR10_num_nodes.png width="60%">

所有节点的邻居数均为8

<img decoding="async" src=CIFAR10_num_neigh.png width="60%">

## SBM dataset

**PATTERN**


每个子图平均有119个节点，不同子图间的节点数差距较大

<img decoding="async" src=PATTERN_num_nodes.png width="60%">

节点的平均邻居数为51，不同节点间邻居数差距较大

<img decoding="async" src=PATTERN_num_neigh.png width="60%">

**CLUSTER**


每个子图平均有117个节点，不同子图间的节点数差距较大

<img decoding="async" src=CLUSTER_num_nodes.png width="60%">

节点的平均邻居数为36，不同节点间邻居数差距较大

<img decoding="async" src=CLUSTER_num_neigh.png width="60%">

## Full-graph dataset

**cora**

每个节点平均有4个邻居，不同节点间邻居数差别较大

<img decoding="async" src=cora_num_neigh.png width="60%">

**arxiv**

每个节点平均有7个邻居，不同节点间邻居数差别很大

<img decoding="async" src=arxiv_num_neigh.png width="60%">

**cite**

每个节点平均有3个邻居，不同节点间邻居数差别不大

<img decoding="async" src=cite_num_neigh.png width="60%">

**pubmed**

每个节点平均有4个邻居，不同节点间邻居数差别较大

<img decoding="async" src=pubmed_num_neigh.png width="60%">