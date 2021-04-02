#pragma once

#include <ATen/core/Tensor.h>

namespace at {
namespace native {

Tensor& mkldnn_copy_(Tensor& self, const Tensor& src, bool non_blocking);
} // namespace native
} // namespace at
