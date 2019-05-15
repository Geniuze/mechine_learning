#!C:\Python27\Python

from __future__ import print_function

import sys

import math
import numpy

import paddle
import paddle.fluid as fluid

param_dirname = "two.model"

def test_reader_func():
    def reader():
        with open("test_2.txt", "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                x,x2,y = line.strip().split("\t")
                yield [[float(x),float(x2)], float(y)]
    return reader

def main():
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
        ] = fluid.io.load_inference_model(param_dirname, infer_exe)

        batch_size = 10

        infer_reader = paddle.batch(
            test_reader_func(), batch_size=batch_size
        )
        infer_data = next(infer_reader())
        infer_feat = numpy.array([data[0] for data in infer_data]).astype("float32")
        infer_label = numpy.array([data[1] for data in infer_data]).astype("float32")

        assert feed_target_names[0] == 'x'

        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: infer_feat},
            fetch_list=fetch_targets
        )

        print("infer results:")
        for idx,val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))
        print("true results:")
        for idx,val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))


if __name__ == "__main__":
    main()
