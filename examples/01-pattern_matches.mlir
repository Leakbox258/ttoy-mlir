module {
  etoy.func private @redundant_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    etoy.return %arg0 : tensor<*xf64>
  }
  etoy.func private @redundant_reshape() -> tensor<*xf64> {
    %0 = etoy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    etoy.return %0 : tensor<2x1xf64>
  }
  etoy.func @main() -> tensor<*xf64> {
    %0 = etoy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = etoy.call @redundant_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
    %2 = etoy.call @redundant_reshape() : () -> tensor<*xf64>
    etoy.return %2 : tensor<*xf64>
  }
}