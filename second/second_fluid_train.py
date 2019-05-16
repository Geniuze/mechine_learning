#!C:\Python27\Python

from __future__ import print_function

import sys

import math
import numpy

import paddle
import paddle.fluid as fluid

param_dirname = "two.model"

def train_reader_func():
    def reader():
        with open("train_2.txt", "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                x,x2,y = line.strip().split('\t')
                yield [[float(x),float(x2)],float(y)]
    return reader

def test_reader_func():
    def reader():
        with open("test_2.txt", "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                x,x2,y = line.strip().split('\t')
                yield [[float(x), float(x2)], float(y)]
    return reader


def main():
    batch_size = 20
    train_reader = paddle.batch(paddle.reader.shuffle(train_reader_func(), buf_size=100), batch_size)

    x = fluid.layers.data(name='x', shape=[2], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.002)
    sgd_optimizer.minimize(avg_loss)

    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(place=place, feed_list=[x,y])

    exe.run(startup_program)

    for _ in range(10) :
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss]
            )
            print("%s, Loss %s" % ("Train", avg_loss_value[0]))

    # save inference model

    fluid.io.save_inference_model(param_dirname, ['x'], [y_predict], exe)

if __name__ == "__main__":
    main()