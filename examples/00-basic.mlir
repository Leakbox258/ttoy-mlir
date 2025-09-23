module {
  ttoy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = ttoy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = ttoy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = ttoy.mul %0, %1 : tensor<*xf64>
    ttoy.return %2 : tensor<*xf64>
  }
  ttoy.func @main() {
    %0 = ttoy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = ttoy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %2 = ttoy.reshape(%1 : tensor<6xf64>) to tensor<2x3xf64>
    %3 = ttoy.call @multiply_transpose(%0, %2) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %4 = ttoy.call @multiply_transpose(%2, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = ttoy.call @multiply_transpose(%3, %4) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %6 = ttoy.call @multiply_transpose(%0, %3) : (tensor<2x3xf64>, tensor<*xf64>) -> tensor<*xf64>
    ttoy.return
  }
}