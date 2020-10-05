


import time
import zmq
import msgpack

import msgpack_numpy as m
m.patch()

import zmq_utils
from zmq_utils import recv_array, recv_pyobj


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# socket2 = context.socket(zmq.REP)
# socket2.bind("tcp://*:5556")

while True:
    #  Wait for next request from client

    print("waiting for request")
    if serialization == "MSGPACK":
        msg_raw = socket.recv()
        start_time = time.time()
        msg = msgpack.unpackb(msg_raw)

        for key, val in msg.items():
            array = zmq_utils.unpack_array(val)
        # img = zmq_utils.unpack_array(msg['img'])

        elapsed_time = time.time() - start_time
        socket.send(b'received message')
        print("unpacking message took:", elapsed_time)
    elif serialization == "MSGPACK-PYTHON":
        msg_raw = socket.recv()
        start_time = time.time()
        msg = msgpack.unpackb(msg_raw)
        elapsed_time = time.time() - start_time
        socket.send(b'received message')
        print("unpacking message took:", elapsed_time)
    elif serialization == "ARRAY":
        img = recv_array(socket)
        socket.send(b'received message')



