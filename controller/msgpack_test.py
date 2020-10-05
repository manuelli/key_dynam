import msgpack
import numpy as np
import time

import zmq_utils

import msgpack_numpy as m
m.patch()

H = 480
W = 640

img = np.array([H,W,3], dtype=np.uint8)
shape = img.shape
dtype = img.dtype

msg = dict()
msg['img'] = zmq_utils.pack_array(img)

print(type(msg['img']['bytes']))
print(type(msg['img']['dtype']))
print(type(msg['img']['shape']))

start_time = time.time()
pack = msgpack.packb(msg)
print("type(pack)", type(pack))
pack_time = time.time() - start_time

start_time = time.time()
msg2 = msgpack.unpackb(pack)
print('msg2.keys()', msg2.keys())
img2 = zmq_utils.unpack_array(msg2['img'])
unpack_time = time.time() - start_time
print("img2.shape")

print("pack time:", pack_time)
print("unpack time:", unpack_time)


msg['img'] = img
start_time = time.time()
pack = msgpack.packb(msg)
print("type(pack)", type(pack))
pack_time = time.time() - start_time

start_time = time.time()
msg2 = msgpack.unpackb(pack)
img2 = msg2['img']
print('img2.shape', img2.shape)
print("img2.dtype", img2.dtype)
unpack_time = time.time() - start_time

print("pack time:", pack_time)
print("unpack time:", unpack_time)
