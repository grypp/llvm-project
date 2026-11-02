//===- DlpackPlugin.cpp - DLPack tensor views for mlir.dsl ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The `_mlirDslDlpack` extension behind the `dlpack` plugin of mlir.dsl. It
// imports any object speaking the DLPack protocol (`__dlpack__`) or the buffer
// protocol through nanobind's ndarray, which owns the producer's lease for as
// long as the view lives, and exposes the tensor metadata to Python. No DLPack
// header or library of its own is involved.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
/// Holds the imported array; nanobind keeps the producer alive through it.
struct TensorView {
  nb::ndarray<nb::ro> array;
};

nb::tuple dims(const nb::ndarray<nb::ro> &array, bool strides) {
  nb::list values;
  for (size_t i = 0, n = array.ndim(); i < n; ++i)
    values.append(strides ? nb::int_(array.stride(i))
                          : nb::int_(array.shape(i)));
  return nb::tuple(values);
}
} // namespace

NB_MODULE(_mlirDslDlpack, m) {
  m.doc() = "DLPack tensor views for mlir.dsl";

  nb::class_<TensorView>(m, "TensorView",
                         "The metadata of a tensor imported through the DLPack "
                         "or buffer protocol; keeps the tensor alive.")
      .def(
          "__init__",
          [](TensorView *self, nb::ndarray<nb::ro> array) {
            new (self) TensorView{std::move(array)};
          },
          "tensor"_a,
          "Imports `tensor`, any object with `__dlpack__` or the buffer "
          "protocol, writable or read-only.")
      .def_prop_ro(
          "data_ptr",
          [](const TensorView &self) {
            return reinterpret_cast<uintptr_t>(self.array.data());
          },
          "The address of the first element (host or device).")
      .def_prop_ro("ndim",
                   [](const TensorView &self) { return self.array.ndim(); })
      .def_prop_ro(
          "shape",
          [](const TensorView &self) { return dims(self.array, false); })
      .def_prop_ro(
          "strides",
          [](const TensorView &self) { return dims(self.array, true); },
          "The strides in elements.")
      .def_prop_ro(
          "dtype_code",
          [](const TensorView &self) { return int(self.array.dtype().code); },
          "The DLPack type code (0 int, 1 uint, 2 float, 4 bfloat, 6 bool, "
          "7 to 17 the narrow floats of DLPack 1.1).")
      .def_prop_ro(
          "dtype_bits",
          [](const TensorView &self) { return int(self.array.dtype().bits); })
      .def_prop_ro(
          "dtype_lanes",
          [](const TensorView &self) { return int(self.array.dtype().lanes); })
      .def_prop_ro(
          "device_type",
          [](const TensorView &self) { return self.array.device_type(); },
          "The DLPack device type (1 CPU, 2 CUDA, 3 CUDA host, 13 CUDA "
          "managed).")
      .def_prop_ro("device_id", [](const TensorView &self) {
        return self.array.device_id();
      });
}
