import numpy as np
import logging

logger = logging.getLogger(__name__)


class MpiArray(object):
    
    # initialize array from a SINGLE source using arr\
    #    set arr and root
    # initialize array from remote sources using local_arr
    #    set local_arr and axis
    def __init__(self, arr=None, local_arr=None, axis=0, padding=0, root=0, comm=None):
        if axis not in (None, 0, 1):
            raise Exception("MpiArray can only scatter on axis 0 or 1, not %s" % str(axis))
        # lazy load mpi4py 
        from mpi4py import MPI
        # initialize variables
        self.comm = comm or MPI.COMM_WORLD # MPI comm, used to send/recv
        self.arr = arr # full array, only stored on root mpi node when not distributed
        self.local_arr = local_arr # local_arr, split along self.axis, stored on every node, with padding
        self.padding = padding
        self.root = root # root mpi_rank that stores full arr
        self.shape = None # shape of the whole ndarray
        self.dtype = None # dtype of ndarray
        # calculate parameters for distributed array or global array
        # TODO: figure out better way to store if we are distributed or not?
        if self.local_arr is not None:
            self.axis = axis
            # calcualte from local_arr sizes along axis
            total_axis_size = self.comm.allreduce(self.local_arr.shape[0], op=MPI.SUM)
            # remove overlap
            total_axis_size -= 2 * padding * (self.mpi_size - 1)
            if self.axis == 0:
                self.shape = (total_axis_size,)+local_arr.shape[axis+1:]
            else:
                self.shape = (local_arr.shape[1], total_axis_size) + local_arr.shape[2:]
            self.dtype = self.local_arr.dtype
        else:
            # take size from root rank that has array (usually zero) 
            self.axis = None
            if self.arr is not None:
                shape = arr.shape
                dtype = arr.dtype
            else:
                shape = None
                dtype = None
            self.shape = self.comm.bcast(shape, root=root)
            self.dtype = self.comm.bcast(dtype, root=root)

    @property
    def mpi_rank(self):
        # rank of this MPI process
        return self.comm.Get_rank()
    
    @property
    def mpi_size(self):
        # total number of MPI processes
        return self.comm.Get_size()

    def is_root(self):
        # returns true if we are the root node
        return self.mpi_rank == self.root

    @property
    def ndim(self):
        # total number of dimensions
        return len(self.shape)
    
    @property
    def itemsize(self):
        # length of data element
        return self.dtype.itemsize

    @property
    def sizes(self):
        return self.split_array_indicies(self.shape, self.mpi_size, self.axis or 0, self.padding)[0]

    @property
    def size(self):
        return self.sizes[self.mpi_rank]

    @property
    def offsets(self):
        return self.split_array_indicies(self.shape, self.mpi_size, self.axis or 0, self.padding)[1]

    @property
    def offset(self):
        return self.offsets[self.mpi_rank]

    @property
    def unpadded_sizes(self):
        return self.split_array_indicies(self.shape, self.mpi_size, self.axis or 0)[0]

    @property
    def unpadded_size(self):
        return self.unpadded_sizes[self.mpi_rank]

    @property
    def unpadded_offsets(self):
        return self.split_array_indicies(self.shape, self.mpi_size, self.axis or 0)[1]

    @property
    def unpadded_offset(self):
        return self.unpadded_offsets[self.mpi_rank]

    @property
    def unpadded_local_arr(self):
        local_offset = self.unpadded_offset - self.offset
        return self.local_arr[local_offset:local_offset + self.unpadded_size]

    @property
    def mpi_dtype(self):
        return self.numpy_to_mpi_dtype(self.dtype)

    @staticmethod
    def numpy_to_mpi_dtype(dtype):
        # lazy load mpi4py 
        from mpi4py import MPI        
        return MPI._typedict[dtype.char]
    
    @staticmethod
    def fromglobalarray(arr, root=0, comm=None):
        return MpiArray(arr, root=root, comm=comm)


    @staticmethod
    def fromlocalarrays(local_arr, axis=0, padding=0, comm=None):
        return MpiArray(local_arr=local_arr, axis=axis, padding=padding, comm=comm)

    # Create a copy of the MpiArray.  This makes a copy of the data as well.
    def copy(self):
        arr = None
        if self.arr is not None:
            arr = self.arr.copy()
        local_arr = None
        if self.local_arr is not None:
            local_arr = self.local_arr.copy()
        return MpiArray(arr, local_arr, self.axis, self.padding, self.root, self.comm)


    # scatter data to MPI nodes
    # axis determines which axis to scatter along
    # returns self.local_arr
    def scatter(self, axis=0, padding=0):
        if axis not in (0,1):
            raise Exception("MpiArray can only scatter on axis 0 or 1, not %s" % str(axis))
        if self.axis is None:
            # scatter data to nodes on axis 0
            self.axis = 0
            self.padding = padding
            local_shape = (self.size,)+self.shape[1:]
            self.local_arr = np.empty(local_shape, dtype=self.dtype)
            #TODO: compare to mpi4py Scatterv
            self._Scatterv(self.sizes, self.offsets)
        if self.axis != axis:
            # swapaxis 0 and 1 in a distributed manner
            self.swapaxes_01(padding)
        elif self.padding != padding:
            # we are distributed on the correct axis, but with the wrong padding
            if padding < self.padding:
                # going to smaller padding, update local_arr directly
                current_offset = self.offset
                self.padding = padding
                relative_offset = self.offset - current_offset
                self.local_arr = self.local_arr[relative_offset:relative_offset+self.size]
            else:
                # going to bigger padding, requires an All2allv
                # calculate overlapping regions for All2allv
                _, new_offsets = self.split_array_indicies(self.shape, self.mpi_size, self.axis, padding)
                send_sizes = np.zeros((self.mpi_size,), dtype=int) # what we are sending
                send_offsets = np.zeros((self.mpi_size,), dtype=int)
                recv_sizes = np.zeros((self.mpi_size,), dtype=int) # what we are receiving
                recv_offsets = np.zeros((self.mpi_size,), dtype=int)
                padding_increase = padding - self.padding
                for recv_rank in range(self.mpi_size):
                    # calculate offset and size for extra padding for each mpi process
                    lower_offset = max(0, self.offsets[recv_rank] - padding_increase)
                    lower_size = self.offsets[recv_rank] - lower_offset
                    upper_offset = self.offsets[recv_rank] + self.sizes[recv_rank]
                    upper_size = min(self.shape[self.axis], upper_offset + padding_increase) - upper_offset 
                    for send_rank in range(self.mpi_size):
                        if send_rank != recv_rank and self.mpi_rank in (recv_rank, send_rank):
                            # now find the overlap for each
                            # The offsets are the global index in the array
                            # we will be sending from the unpadded_local_array
                            # we will be receiving into the new expanded local_arr, offset will be zero
                            unpadded_offset_send = self.unpadded_offsets[send_rank]
                            unpadded_size_send = self.unpadded_sizes[send_rank]
                            for i, (offset, size) in enumerate(((lower_offset, lower_size), (upper_offset, upper_size))):
                                if offset < unpadded_offset_send + unpadded_size_send and \
                                   offset + size > unpadded_offset_send:
                                    # send_rank has data to send to recv_rank
                                    # store offset/size for both
                                    overlap_size = min(offset + size, unpadded_offset_send + unpadded_size_send) - \
                                                   max(offset, unpadded_offset_send)
                                    if self.mpi_rank == send_rank:
                                        send_sizes[recv_rank] = overlap_size
                                        send_offsets[recv_rank] = max(0, offset - unpadded_offset_send)
                                    if self.mpi_rank == recv_rank:
                                        recv_sizes[send_rank] = overlap_size
                                        if i == 0:
                                            # lower limit, recv into bottom of local_arr
                                            recv_offsets[send_rank] = max(0, unpadded_offset_send - offset)
                                        else:
                                            # upper limit, we need to recieve this into the upper part of local_arr
                                            recv_offsets[send_rank] = max(offset, unpadded_offset_send) - new_offsets[recv_rank]
                # now that we have the send and recv sizes/offsets calculated
                # create a new array that will replace local_arr
                local_arr_recv = np.empty(self._calc_local_arr_shape(padding=padding), dtype=self.dtype)
                # mpi4py treats ndarrays as flat, so we need a stride for the other dimensions
                stride = np.prod(self.local_arr.shape[1:])
                self.comm.Alltoallv([self.unpadded_local_arr, send_sizes*stride, send_offsets*stride, self.mpi_dtype],
                                    [local_arr_recv, recv_sizes*stride, recv_offsets*stride, self.mpi_dtype])
                # copy local_arr portion
                local_offset = self.offset - new_offsets[self.mpi_rank]
                local_arr_recv[local_offset:local_offset+self.local_arr.shape[0]] = self.local_arr[:]
                self.local_arr = local_arr_recv
                self.padding = padding
        return self.local_arr


    # replacement for the mpi4py ScatterV, which was slow for me
    # TODO: use same syntax as mpi4py
    def _Scatterv(self, sizes, offsets):
        # lazy load mpi4py 
        from mpi4py import MPI        
        # send to all nodes
        if self.is_root():
            reqs = []
            for i in range(self.mpi_size):
                if i != self.root:
                    data = self.arr[offsets[i]:offsets[i]+sizes[i]]
                    reqs.append(self.comm.Isend(data, i))
            self.local_arr = self.arr[offsets[self.root]:offsets[self.root]+sizes[self.root]]
            MPI.Request.Waitall(reqs)
        else:
            self.comm.Recv(self.local_arr, source=self.root)
    
    
    # gather data from MPI nodes
    # the root mpi rank receives the actual array (this changes root for the MpiArray)
    # the other MPI processes return None
    # if axis == 0, data returned in original order
    # if axis == 1, data should be returned after a swapaxes (0,1) has been applied
    def gather(self, axis=0, root=0, delete_local=False):
        if axis not in (None, 0, 1):
            raise Exception("MpiArray can only gather on axis 0 or 1, not %s" % str(axis))
        elif self.axis is None:
            # array hasn't been distributed, check axis
            if self.root == root:
                # array already in correct MPI process
                if axis == 0:
                    # array already stored correctly!
                    return self.arr
                else:
                    # array needs swapaxes, do it distributed
                    self.scatter(axis)
                    # we'll gather below
            else:
                # array not in correct MPI process
                if axis == 0:
                    if self.is_root():
                        # send data to new root node
                        self.comm.Send(self.arr, dest=root)
                        self.arr = None
                    elif self.mpi_rank == root:
                        # get data from old root node
                        self.arr = np.empty(self.shape, dtype=self.dtype)
                        self.comm.Recv(self.arr, source=self.root)
                    self.root = root
                    return self.arr
                else:
                    # swapaxes and transfer to new root, 
                    # scatter to axis 1, then gather
                    self.scatter(axis)
                    self.arr = None
                # gather below
        # at this point self.axis should not be None 
        # swap axis if needed
        if self.axis != axis:
            self.swapaxes_01()
        # we now just need to gather the data in self.arr on the root
        self.root = root
        if self.is_root():
            if self.axis == 0:
                arr_shape = self.shape
            else:
                arr_shape = (self.shape[1], self.shape[0]) + self.shape[2:] 
            if self.arr is None or self.arr.shape != arr_shape:
                self.arr = np.empty(arr_shape, dtype=self.dtype)
        else:
            # make sure we don't have extra copies of the dataset on other nodes
            self.arr = None
        self._Gatherv()
        
        if delete_local:
            self.delete_local()
        return self.arr


    def _Gatherv(self):
        # lazy load mpi4py 
        from mpi4py import MPI
        # all nodes send back to root
        if self.is_root():
            reqs = []
            for i in range(self.mpi_size):
                if i != self.root:
                    data = self.arr[self.unpadded_offsets[i]:self.unpadded_offsets[i]+self.unpadded_sizes[i]]
                    reqs.append(self.comm.Irecv(data, source=i))
            self.arr[self.unpadded_offset:self.unpadded_offset+self.unpadded_size] = self.unpadded_local_arr[:]
            MPI.Request.Waitall(reqs)
        else:
            self.comm.Send(self.unpadded_local_arr, dest=0)
        return self.arr
        

    # Do a distributed swap of axes 0 and 1
    # Equivalent to the follow, except it does it in a distributed manner
    # mpiarray.gather()
    # if mpiarray.mpi_rank == 0:
    #     np.swapaxes(mpiarray.arr, 0, 1)
    # mpiarray.scatter()
    # NOTE: must already be scattered to work.
    def swapaxes_01(self, padding=0):
        if self.axis not in (0, 1):
            raise Exception("Array must already be scattered along axis 0 or 1 for swapaxes_01, not %s" % str(self.axis))
        # calculate the shape of the whole array after swapaxes
        if self.axis == 0:
            # we are switching to axis = 1
            new_shape = (self.shape[1], self.shape[0])+self.shape[2:]
        else:
            # we are switching to axis = 0
            new_shape = self.shape
        
        # planned distribution of data (includes padding)
        sizes, offsets = self.split_array_indicies(new_shape, self.mpi_size, padding=padding)
#         logger.debug("%d: sizes=%s" % (self.mpi_rank, str(self.sizes)))
#         logger.debug("%d: offsets=%s" % (self.mpi_rank, str(self.offsets)))
#         logger.debug("%d: unpadded_local_shape=%s" % (self.mpi_rank, str(self.unpadded_local_arr.shape)))

        # swap axes for sending local data and require alignment
        # NOTE: will create a copy.
        # NOTE: must preserve existing padding at this point
        swapped_local_arr = np.require(np.swapaxes(self.unpadded_local_arr, 0, 1), requirements='C')
        self.local_arr = None # no longer needed, save space
        # create flat array for recv data
        recv_stride = np.prod(new_shape[1:])
        local_arr_recv = np.empty(sizes[self.mpi_rank]*recv_stride, dtype=self.dtype)
        # calculate where to send data
        swapped_sizes, swapped_offsets = self.split_array_indicies(swapped_local_arr.shape, self.mpi_size, padding=padding)
        # calculate data being received in units of elements (recv is flat array)
        self.padding = padding
        recv_sizes = np.empty(self.mpi_size, dtype=np.int)
        for i in range(self.mpi_size):
            recv_sizes[i] = sizes[self.mpi_rank] * self.unpadded_sizes[i] * np.prod(new_shape[2:])
        recv_offsets = np.zeros(self.mpi_size, dtype=np.int)
        recv_offsets[1:] = np.cumsum(recv_sizes)[:-1]
    
#         logger.debug("%d: swapped_local_arr.shape=%s"%(self.mpi_rank, str(swapped_local_arr.shape)))
#         logger.debug("%d: local_arr_recv.shape=%s"%(self.mpi_rank, str(local_arr_recv.shape)))
#         logger.debug("%d: swapped: %s, %s" % (self.mpi_rank, str(swapped_sizes), str(swapped_offsets)))
#         logger.debug("%d: recv: %s, %s" % (self.mpi_rank, str(recv_sizes), str(recv_offsets)))
        
        # send and receive data
        swapped_stride = np.prod(swapped_local_arr.shape[1:])
        self.comm.Alltoallv([swapped_local_arr, swapped_sizes*swapped_stride, swapped_offsets*swapped_stride, self.mpi_dtype],
                            [local_arr_recv, recv_sizes, recv_offsets, self.mpi_dtype])
        # delete sending array
        del swapped_local_arr
        # create new array to store data
        self.local_arr = np.empty((sizes[self.mpi_rank],) + new_shape[1:], dtype=self.dtype)
        # now do local copies to fix data arrangement
#         logger.debug("local_arr.shape="+str(self.local_arr.shape))
#         logger.debug("local_arr_recv.shape="+str(local_arr_recv.shape))
        for i in range(self.mpi_size):
            self.local_arr[:, self.unpadded_offsets[i]:self.unpadded_offsets[i]+self.unpadded_sizes[i]] = local_arr_recv[recv_offsets[i]:recv_offsets[i]+recv_sizes[i]].reshape((self.local_arr.shape[0], self.unpadded_sizes[i])+self.local_arr.shape[2:])
        del local_arr_recv
        self.axis ^= 1 # switched axis


    # delete local arrays
    def delete_local(self):
        self.local_arr = None
        self.padding = 0
        self.axis = None


    # delete global array
    def delete_global(self):
        self.arr = None

    
    # calculate local_arr shape
    def _calc_local_arr_shape(self, shape=None, axis=None, padding=None):
        if shape is None:
            shape = self.shape
        if axis is None:
            axis = self.axis
        if padding is None:
            padding = self.padding
        sizes, _ = self.split_array_indicies(shape, self.mpi_size, axis, padding)
        if axis == 0:
            return (sizes[self.mpi_rank], ) + shape[1:]
        else:
            return (sizes[self.mpi_rank], shape[0]) + shape[2:]

    
    # Calculate offsets and indicies to split an array 
    # to send/recv from all of the MPI nodes
    @staticmethod
    def split_array_indicies(shape, mpi_size, axis=0, padding=0):
        # nodes calculate offsets and sizes for sharing
        chunk_size = shape[axis] // mpi_size
        leftover = shape[axis] % mpi_size
        sizes = np.ones(mpi_size, dtype=np.int) * chunk_size
        # evenly distribute leftover across workers
        # NOTE: currently doesn't add leftover to rank 0, 
        # since rank 0 usually has extra work to perform already
        sizes[1:leftover+1] += 1
        offsets = np.zeros(mpi_size, dtype=np.int)
        offsets[1:] = np.cumsum(sizes)[:-1]
        # now compensate for padding
        upper_limit = offsets + sizes + padding
        upper_limit[upper_limit > shape[axis]] = shape[axis]
        offsets -= padding
        offsets[offsets<0] = 0
        sizes = upper_limit - offsets
        return sizes, offsets
