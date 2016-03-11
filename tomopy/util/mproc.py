#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module for multiprocessing tasks.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from mpi4py import MPI
import numpy as np
import math
import logging
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['distribute_jobs']


def dtype2mpi(dtype):
    return MPI._typedict[dtype.char]
 
 
def size_from_shape(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size

# scatter numpy array to all nodes
#TODO: leave fewest jobs for root node (since needs to do communication)
#TODO: do a better job of evening the number of slices per rank
def scatter_arr(arr, root=0):#, overlap=0):
    # first send every node the shape and dtype
    if rank == root:
        shape = arr.shape
        dtype = arr.dtype
    else:
        shape = None
        dtype = None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    # nodes calculate offsets and sizes for sharing
    chunk_size = int(math.ceil(shape[0]/size))
    offsets = [chunk_size * r for r in range(size)]
    sizes = [chunk_size for _ in range(size)]
    # remove remainder from last slice
    sizes[size-1] -= chunk_size * size - shape[0]
    local_shape = (sizes[rank],)+shape[1:]
    local_arr = np.empty(local_shape, dtype=dtype)

    # send to all nodes
    if rank == root:
        reqs = []
        for i in range(size):
            if i != root:
                data = arr[offsets[i]:offsets[i]+sizes[i]]
                reqs.append(comm.Isend(data, i))
        local_arr = arr[offsets[root]:offsets[root]+sizes[root]]
        MPI.Request.Waitall(reqs)
    else:
        comm.Recv(local_arr, source=root)

    print("%d: %s"%(rank, str(local_arr.shape)))
    return local_arr


# gather numpy array from all nodes
#TODO: leave fewest jobs for root node (since needs to do communication)
#TODO: do a better job of evening the number of slices per rank
def gather_arr(arr, local_arr, root=0):#, overlap=0):
    # first send every node the shape and dtype
    if rank == root:
        shape = arr.shape
    else:
        shape = None
    shape = comm.bcast(shape, root=root)
    # nodes calculate offsets and sizes for sharing
    chunk_size = int(math.ceil(shape[0]/size))
    sizes = [chunk_size for _ in range(size)]
    offsets = [chunk_size * r for r in range(size)]
    # remove remainder from last slice
    sizes[size-1] -= shape[0] - chunk_size * size
    
    # all nodes send back to root
    if rank == root:
        reqs = []
        for i in range(size):
            if i != root:
                data = arr[offsets[i]:offsets[i]+sizes[i]]
                reqs.append(comm.Irecv(data, source=i))
        arr[offsets[root]:offsets[root]+sizes[root]] = local_arr[:]
        MPI.Request.Waitall(reqs)
    else:
        comm.Send(local_arr, dest=root)


def distribute_jobs(arr,
                    func,
                    axis,
                    args=None,
                    kwargs=None,
                    ncore=None,
                    nchunk=None,
                    out=None):
    """
    Distribute N-dimensional shared-memory array into cores by splitting along
    an axis.

    Parameters
    ----------
    arr : ndarray, or iterable(ndarray)
        Array(s) to be split up for processing.
    func : func
        Function to be parallelized.  Should return an ndarray.
    args : list
        Arguments of the function in a list.
    kwargs : list
        Keyword arguments of the function in a dictionary.
    axis : int
        Axis along which parallelization is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size to use when parallelizing data.  None will maximize the chunk
        size for the number of cores used.  Zero will use a chunk size of one, 
        but will also remove the dimension from the array.
    out : ndarray, optional
        Output array.  Results of functions will be compiled into this array.
        If not provided, last arr will be used for output.

    Returns
    -------
    ndarray
        Output array.
    """
    # parameters needed for MPI calls
    arrs = None
    num_arrs = None
    
    if rank == 0:
        if isinstance(arr, np.ndarray):
            arrs = [arr]
        else:
            # assume this is multiple arrays
            arrs = list(arr)

        num_arrs = len(arrs)
    num_arrs = comm.bcast(num_arrs)

    if arrs is None:
        arrs = [None] * num_arrs

    # prepare all args (func, args, kwargs)
    # NOTE: args will include shared_arr slice as first arg
    args = args or tuple()
    kwargs = kwargs or dict()

    #TODO: decide if parameters always need to be shared
    args = comm.bcast(args)
    kwargs = comm.bcast(kwargs)

    # distribute arrs to all nodes
    start = time.time()
    local_arrs = []
    for i in range(num_arrs):
        local_arrs.append(scatter_arr(arrs[i]))
    print("%d: scatter took: %0.2f s" % (rank, time.time() - start))

    # run function
    local_out = None
    if nchunk != 0:
        result = func(*(local_arrs + list(args)), **kwargs)
        if result is not None:
            local_out = result
    else:
        axis_size = local_arrs[0].shape[0]
        for i in range(axis_size):
            flat_arrs = [local_arr[i] for local_arr in local_arrs]
            result = func(*(flat_arrs + list(args)), **kwargs)
            if result is not None and not np.may_share_memory(result, local_arrs[-1]):
                if local_out is None:
                    local_out = np.empty((axis_size,)+result.shape, dtype=result.dtype)
                local_out[i] = result[:]
    
    # gather result
    if rank == 0 and out is None:
        out = arrs[-1]
    
    start = time.time()
    if local_out is None:
        local_out = local_arrs[-1]
    gather_arr(out, local_out)
    print("%d: gather took: %0.2f s" % (rank, time.time() - start))
    return out

