

import msgpack_numpy as m
m.patch()

from key_dynam.controller.zmq_utils import ZMQServer


zmq_server = ZMQServer()

while True:
    print("waiting to receive data from client")
    msg = zmq_server.recv_data()

    print("msg", msg)

    print("msg.keys()", msg.keys())
    print("msg['type']", msg['type'])

    # print("rgb shape", msg['data']['observations']['images']['camera']['d415_01']['rgb'].shape)

    resp = {'type': 'NORMAL'}
    zmq_server.send_data(resp)



