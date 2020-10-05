from key_dynam.controller.zmq_utils import convert

msg = {'lucas': 'PLAN'}
msg['data'] = {'pos': [0,1,2]}


msg_convert = convert(msg)
print("msg_convert", msg_convert)


def func(a):
    return a

a = map(func, [0,1,2])
print(type(a))
print(list(a))