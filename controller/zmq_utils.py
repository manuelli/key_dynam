import zmq
import pickle
import numpy
import msgpack
import collections

# enable msgpack_numpy
import msgpack_numpy as m
m.patch()

def send_pyobj(socket, data):
    pickle_data = pickle.dumps(data)
    socket.send(pickle_data)
    # socket.send_pyobj(data)

def recv_pyobj(socket):
    pickle_data = socket.recv()
    data = pickle.loads(pickle_data, encoding='latin1')
    return data

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def pack_array(A):
    return {'bytes': A.tobytes(),
            'dtype': str(A.dtype),
            'shape': A.shape}

def unpack_array(data):
    buf = memoryview(data['bytes'])
    A = numpy.frombuffer(buf, dtype=data['dtype'])
    return A.reshape(data['shape'])


# def convert(data):
#     if isinstance(data, bytes):
#         return data.decode('ascii')
#     elif isinstance(data, dict):
#         return dict(map(convert, data.items()))
#     elif isinstance(data, tuple):
#         return map(convert, data)
#     else:
#         return data

# def convert(data):
#     """
#     Converts all bytes to strings with ascii decoding
#     Recursively
#     """
#
#     if isinstance(data, bytes):
#         print('\n')
#         print("bytes")
#         # print("data", data)
#         return_data = data.decode('ascii')
#         print("return_data", return_data)
#         return return_data
#     elif isinstance(data, collections.Mapping):
#         print('\n')
#         print("collections.Mapping")
#         # print("type(data)", type(data))
#         return_data = type(data)(map(convert, data.items()))
#         print("data", data)
#         print("return_data", return_data)
#     elif isinstance(data, collections.Iterable):
#         print('\n')
#         print("collections.Iterable")
#         # print("data", data)
#         return_data = type(data)(map(convert, data))
#         print("data", data)
#         print("return_data", return_data)
#         return return_data
#     else:
#         return data

def convert(data):
  if isinstance(data, bytes):
        return data.decode()
  elif isinstance(data, (str, int)):
        return str(data)
  elif isinstance(data, dict):
        return dict(map(convert, data.items()))
  elif isinstance(data, tuple):
        return tuple(map(convert, data))
  elif isinstance(data, list):
        return list(map(convert, data))
  elif isinstance(data, set):
      return set(map(convert, data))
  else:
      return data


class ZMQServer(object):
    """
    Wrapper for ZMQ server
    """

    def __init__(self, port=5555):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://*:%d" %(port))

    def send_data(self,
                  data, # dict, only non-primitive type is numpy.array
                  ):
        """
        Send data to server over ZMQServer
        :param data:
        :return:
        """
        self._socket.send(msgpack.packb(data))

    def recv_data(self):
        """
        Receives the response from the server over socket
        Returns the data
        :return:
        """

        data_raw = self._socket.recv()
        data = msgpack.unpackb(data_raw)
        data = convert(data)
        return data

