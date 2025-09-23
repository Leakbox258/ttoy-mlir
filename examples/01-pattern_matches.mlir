module {
  ttoy.func private @redundant_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    ttoy.return %arg0 : tensor<*xf64>
  }
  ttoy.func private @redundant_reshape() -> tensor<*xf64> {
    %0 = ttoy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    ttoy.return %0 : tensor<2x1xf64>
  }
  ttoy.func @main() -> tensor<*xf64> {
    %0 = ttoy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = ttoy.call @redundant_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
    %2 = ttoy.call @redundant_reshape() : () -> tensor<*xf64>
    ttoy.return %2 : tensor<*xf64>
  }
}