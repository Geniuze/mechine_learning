#!C:\Python27\Python
# -*- coding=UTF-8 -*-

from __future__ import print_function

import sys

import math
import numpy

import paddle
import paddle.fluid as fluid

def train_reader_func():
    def reader():
        with open("train.txt", "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                x,y = line.strip().split('\t')
                yield [[float(x),], float(y)]
    return reader

def test_reader_func():
    def reader():
        with open("test.txt", "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                x,y = line.strip().split('\t')
                yield [[float(x),], float(y)]
    return reader

def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        print ("accumulated:", accumulated)
        print ("outs:", outs)
        print(zip(accumulated, outs))
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
        exit()
    return [x_d / count for x_d in accumulated]

def save_result(points1, points2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('.image/prediction_gt.png')

def main():
    batch_size = 20
    train_reader = paddle.batch(
        paddle.reader.shuffle(train_reader_func(), buf_size=100),
        batch_size=batch_size
    )
    test_reader = paddle.batch(
        paddle.reader.shuffle(test_reader_func(), buf_size=100),
        batch_size=batch_size
    )

    #x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    sgd_optmizer = fluid.optimizer.SGD(learning_rate=0.02)
    sgd_optmizer.minimize(avg_loss)

    test_program = main_program.clone(for_test=True)

    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    params_dirname = "one.model"
    num_epochs = 10

    feeder = fluid.DataFeeder(place=place, feed_list=[x,y])
    exe.run(startup_program)

    train_prompt = "Train cost"
    test_prompt = "Test cost"
    step = 0

    exe_test = fluid.Executor(place)

    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss])
            if step % 10 == 0:
                print("%s, Step %d, Cost %f" %
                    (train_prompt, step, avg_loss_value[0]))
            if step % 100 == 0:
                test_metics = train_test(
                    executor=exe_test,
                    program=test_program,
                    reader=test_reader,
                    fetch_list=[avg_loss],
                    feeder=feeder)
                print("%s, Step %d, Cost %f" %
                    (test_prompt, step, test_metics[0]))
                if test_metics[0] < 0.00001:
                    break
            step += 1
            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NAN loss, training failed.")
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
        ] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10 

        infer_reader = paddle.batch(
            test_reader_func(), batch_size=batch_size)
        infer_data = next(infer_reader())
        infer_feat = numpy.array(
            [data[0] for data in infer_data]
        ).astype("float32")
        infer_label = numpy.array(
            [data[1] for data in infer_data]
        ).astype("float32")

        assert feed_target_names[0] == 'x'
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: infer_feat},
            fetch_list=fetch_targets)

        print("infer results: ")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))
        
        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))

        save_result(results[0], infer_label)


if __name__ == '__main__':
    main()
