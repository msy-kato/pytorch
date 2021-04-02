#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/mkldnn/Copy.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(false, "copy_mkldnn_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "copy_mkldnn_: only support same size tensor.");
  ideep::tensor& x = itensor_from_mkldnn(src);
  ideep::tensor& y = itensor_from_mkldnn(self);
  ideep::direct_copy::compute(x, y);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
