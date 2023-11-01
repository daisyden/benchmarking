# Introduction
This document will introduce how to run generative model with multiple ranks and how to do scaling analysis based on ipex, deepspeed and oneCCL. The cluster as example is SPR cluster based on CVL nic.
(This guide is verified on ipex cpu_deive branch commit 78c78d9909ebfd5e64f887ce60cbb523e680e3f6)


# Preparation
## Get system information
Before data collection we need to understand the system and network configurations.

```
# Check cpu and numa configurations
lscpu
numactl -H

# Get irdma devices 
ibv_devices 

# Check the numa node of each nic, for example if your ibv_devices report a nic named irdma-cvl12tf2
cat /sys/class/infiniband/irdma-cvl12tf2/device/numa_node  

# Check libfabric libraries
ls /usr/lib64/libfabric.so.*
ls /usr/lib64/libfabric/libpsm3-fi.so

# Check memory capacity
free -H

```
## NOPASSWD ssh
To run multinode test please ensure your node can access by no password ssh.  
```
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_rsa.pub user@ipaddr2

#ensure the following file permission is the same on both machines, otherwise use chmod to update the permission
ls -l ~/.ssh/
total 28
-rw-------. 1 sdp sdp 4005 Oct 19 18:21 authorized_keys
-rwxr-xr-x  1 sdp sdp  136 Sep 15 19:15 config
-rw-------. 1 sdp sdp 2622 Oct 19 18:35 id_rsa
-rw-r--r--. 1 sdp sdp  589 Oct 19 18:35 id_rsa.pub
-rw-r--r--. 1 sdp sdp 9561 Oct 19 18:29 known_hosts
```

Then you can try "ssh ipaddr2" without password.

## Build enviroment
Please follow README.md under ipex cpu-device/examples/cpu/inference/python/llm to build enviroment, suppose your conda env is llm.

# Data collection
## Single rank
Please follow README.md to run single rank test.

## Single node with multiple ranks
Recommend to use SHM based low latency allreduce implemented in deepspeed for single node communication, it supports up to 8 ranks.
```
unset KMP_AFFINITY
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank --bind_core_list 0-$((OMP_NUM_THREADS - 1)),56-$((56 + OMP_NUM_THREADS - 1)) distributed/run_generation_with_deepspeed.py  --model-id $model_id --dytpe bfloat16 --ipex   --batch-size 1 --benchmark --max-new-tokens ${output} --input-tokens ${input} --token-latency --num-iter ${num_iter} --num-warmup ${warmup} 

```

## Multiple node

Suppose we will run 4 ranks on 2 nodes, each node starts two ranks. Create a nodefile:
```
node1 slots=2
node2 slots=2
```

```
WORKDIR=`pwd`
export PATH=~/miniconda3/bin:$PATH
PREFIX=llm
ONECCL_DIR=${WORKDIR}/oneCCL
source activate $PREFIX
source ~/miniconda3/envs/$PREFIX/lib/python3.9/site-packages/oneccl_bind_pt-2.1.0+cpu-py3.9-linux-x86_64.egg/oneccl_bindings_for_pytorch/env/setvars.sh

OMP_NUM_THREADS=48 # On a 56c socket system, leave some cores for CCL worker and leave some cores idle to reduce stragger effect

# Config OneCCL 
function nw_config()
{
    worker=$1
    if [ "$worker" = "1" ];then
        export CCL_WORKER_AFFINITY="167,223"    #Worker affinity, it is recommended to avoid a conflict with computation cores
        export CCL_WORKER_COUNT=1               #CCL worker thread 
    fi

    if [ "$worker" = "2" ];then
        export CCL_WORKER_AFFINITY='166,167,222,223'
        export CCL_WORKER_COUNT=2
    fi

    if [ "$worker" = "4" ];then
        export CCL_WORKER_AFFINITY="164,165,166,167,220,221,222,223"
        export CCL_WORKER_COUNT=4
    fi

    if [ "$worker" = "8" ];then
        export CCL_WORKER_AFFINITY='160,161,162,163,164,165,166,167,196,197,198,199,220,221,222,223'
        export CCL_WORKER_COUNT=8
    fi

    export CCL_ALLREDUCE=rabenseifner # Other algorithms inlcude nreduce, ring and recursive_doubling. Rabenseifner algorithm is more friendly for latency sensitive workload

    export CCL_BF16=avx512bf
    #export CCL_LOG_LEVEL=info
    export CCL_ATL_TRANSPORT=mpi #Other option is ofi

    # Copy your local libfabirc to torch-ccl
    cp /usr/lib64/libfabric.so.1.18.1  ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bind_pt-2.1.0+cpu-py3.9-linux-x86_64.egg/oneccl_bindings_for_pytorch/lib/libfabric.so.1
    cp /usr/lib64/libfabric/libpsm3-fi.so ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bind_pt-2.1.0+cpu-py3.9-linux-x86_64.egg/oneccl_bindings_for_pytorch/lib/prov/libpsm3-fi.so

}

# Create mpi argments
# Assume your ibv_devices output has 4 nics: irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2
function build_launch_args(){
    margs="--genv CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"
    margs="$margs --genv CCL_MNIC=local"                  # Select all NICs local for the NUMA node that corresponds to process pinning
    margs="$margs --genv CCL_MNIC_COUNT=2"                # The maximum number of NICs that should be selected for oneCCL workers. 
    margs="$margs --genv CCL_MNIC_NAME='irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2'"  # to control multi-NIC selection by NIC names
    margs="$margs --genv CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY}"
    margs="$margs --genv CCL_ATL_TRANSPORT=$CCL_ATL_TRANSPORT"   # Select the transport for inter-process communications
    margs="$margs --genv PSM3_ALLOW_ROUTERS=1"                   # Consider all endpoints accessible, even if they have different IPv4 subnets    
    margs="$margs --genv PSM3_RDMA=1"                            # Use Rendezvous module for node-to-node level RC QPs for Rendezvous
    margs="$margs --genv PSM3_RV_MR_CACHE_SIZE=8192"             
    margs="$margs --genv FI_PROVIDER_PATH=/usr/lib64/libfabric"  # Specify the location of the installed PSM3 provider, when use torch-ccl the version in torch-ccl enviroment will be used
    margs="$margs --genv PSM3_NIC_SPEED=100000"                  # Avoid to use Mallanox for rdma  
    margs="$margs --genv PSM3_KASSIST_MODE=none"                 
    margs="$margs --genv PSM3_NIC=irdma-cvl*tf2"                 # Specifies the Device Unit number or RDMA device name (as shown in ibv_devices).
    margs="$margs --genv PSM3_MULTI_EP=1"                        # Enables more than one PSM3 endpoint to be opened in a process
    margs="$margs --genv FI_PROVIDER=psm3"                       # Ensure the Intel® PSM3 OFI provider is used


    # PSM3_DEVICES enables one or more of the following devices for communication
    # self: allows a process to send messages to other processes on the same host via linux shared memory; 
    # nic: allows a process to send messages to processes on other hosts
    # shm: allows a process to send messages to other processes on the same host via linux shared memory. 
    # When we start multiple ranks for single node we could specify "self,nic" to avoid communication with SHM, and specify "self,shm" to only use SHM for communicaiton. 
    #margs="$margs --genv PSM3_DEVICES=\'self,nic\'"
}

# Run 
export LD_LIBRARY_PATH=${ONECCL_DIR}/build/_install/lib:${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bind_pt-2.1.0+cpu-py3.9-linux-x86_64.egg/oneccl_bindings_for_pytorch/lib/:${LD_LIBRARY_PATH}
export MASTER_ADDR=$(hostname)
nw_config 4                          # You could tune the worker number for your workload
build_launch_args

# For example to run bf16
deepspeed --no_ssh_check --hostfile=nodefile  --bind_cores_to_rank --bind_core_list 0-$((OMP_NUM_THREADS - 1)),56-$((56 + OMP_NUM_THREADS - 1)) --launcher impi --launcher_args "--genv LD_LIBRARY_PATH=${LD_LIBRARY_PATH} $margs -l " distributed/run_generation_with_deepspeed.py  --model-id $model_id --dtype bfloat16 --ipex  --batch-size 1 --benchmark --max-new-tokens ${output} --input-tokens ${input} --token-latency --num-iter ${num_iter} --num-warmup ${warmup} 
 
```

