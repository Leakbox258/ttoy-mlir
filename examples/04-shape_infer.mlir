module {
  ttoy.func @main() -> tensor<*xf64> {
    %0 = ttoy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ttoy.print %0 : tensor<2x3xf64>
    %1 = ttoy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    ttoy.return %1 : tensor<2x1xf64>
  }
}