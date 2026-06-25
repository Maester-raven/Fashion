#!/usr/bin/env python3
"""Build a static TensorRT engine from the shared RTMDet ONNX graph."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = Path(args.onnx).read_bytes()
    if not parser.parse(onnx_bytes):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError("TensorRT failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * 1024**3)
    )
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This platform does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"TensorRT={trt.__version__} fp16={args.fp16}")
    print(f"inputs={network.num_inputs} outputs={network.num_outputs}")
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        print("INPUT", tensor.name, tuple(tensor.shape), tensor.dtype)
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        print("OUTPUT", tensor.name, tuple(tensor.shape), tensor.dtype)

    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    output_path = Path(args.engine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialized)
    print(
        f"ENGINE_OK path={output_path} size_bytes={output_path.stat().st_size} "
        f"build_seconds={time.time() - start:.1f}"
    )


if __name__ == "__main__":
    main()
