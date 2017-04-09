"""
    communicate.py
    Nicholas S. Bradford
    4/7/2017

"""

import zmq


class CommunicationZMQ():


    def __init__(self):
        port = "5555"
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.connect("tcp://127.0.0.1:%s" % port)


    def sendMessage(self, protobuf_msg):
        self.socket.send(protobuf_msg.SerializeToString())
