#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2020  Microsoft (author: Ke Wang)

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import re

import torch.distributed as dist


LOG = logging.getLogger(__name__)


def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)


def ompi_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)


def ompi_local_size():
    """Find OMPI local size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE') or 1)


def get_init_method_philly():
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    master_ip = None
    if my_rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_ip = result.decode('utf-8').split()[0]
    master_ip = comm.bcast(master_ip, root=0)
    LOG.info(f"get_master_ip - {master_ip}")

    start_port_range = int(os.environ['PHILLY_CONTAINER_PORT_RANGE_START'])
    end_port_range = int(os.environ['PHILLY_CONTAINER_PORT_RANGE_END'])
    torch_port = start_port_range + (end_port_range - start_port_range) // 2
    master_port = comm.bcast(torch_port, root=0)

    return f'tcp://{master_ip}:{master_port}'


def get_init_method_amlk8s():
    regexp = '[\s\S]*export[\s]*DLTS_SD_worker0_IP=([0-9.]+)[\s|s]*'
    with open('/dlts-runtime/env/init.env', 'r') as f:
        line = f.read()
    match = re.match(regexp, line)
    if match:
        ip = str(match.group(1))
        LOG.info(f'master node ip is {ip}')
        return f'tcp://{ip}:6000'
    else:
        raise ValueError('did not find master node ip')
    return f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"


def get_init_method_aml():
    # AZ_BATCHAI_MPI_MASTER_NODE or AZ_BATCHAI_JOB_MASTER_NODE_IP
    # "AZ_BATCH_MASTER_NODE": "10.0.0.6:6000" # only on multi node

    if 'AZ_BATCH_MASTER_NODE' in os.environ:
        return 'tcp://' + os.environ.get('AZ_BATCH_MASTER_NODE')

    master_ip = os.environ.get('AZ_BATCHAI_JOB_MASTER_NODE_IP')
    return f'tcp://{master_ip}:6000'


def get_init_method():
    if 'PHILLY_RUNTIME_CONFIG' in os.environ:
        return get_init_method_philly()
    else:
        return get_init_method_amlk8s()


def local_init(backend='nccl'):
    """Local init"""
    master_ip = '127.0.1.1'
    master_port = '50065'
    init_method = f'tcp://{master_ip}:{master_port}'
    world_size = ompi_size()
    rank = ompi_rank()
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    LOG.info(
        f'Init Method: {init_method}, World Size: {world_size}, '
        f'Backend: {backend}, rank: {rank}'
    )


def dist_init(backend='nccl'):
    """Init torch distributed using TCP"""

    LOG.debug('Initializing torch distributed')


    if dist.is_initialized():
        LOG.info('Torch distributed already initialized!')
        return

    if os.environ.get('PHILLY_RUNTIME_CONFIG') is None and \
            not os.path.isfile('/dlts-runtime/env/init.env'):
        local_init()
        return

    init_method = get_init_method()
    world_size = ompi_size()
    rank = ompi_rank()
    LOG.info(
        f'Init Method: {init_method}, World Size: {world_size}, '
        f'Backend: {backend}, rank: {rank}'
    )
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    LOG.debug('init_process_group done!')
