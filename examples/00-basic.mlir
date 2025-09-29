module {
  etoy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = etoy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = etoy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = etoy.mul %0, %1 : tensor<*xf64>
    etoy.return %2 : tensor<*xf64>
  }
  etoy.func @main() {
    %0 = etoy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = etoy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %2 = etoy.reshape(%1 : tensor<6xf64>) to tensor<2x3xf64>
    %3 = etoy.call @multiply_transpose(%0, %2) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %4 = etoy.call @multiply_transpose(%2, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = etoy.call @multiply_transpose(%3, %4) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %6 = etoy.call @multiply_transpose(%0, %3) : (tensor<2x3xf64>, tensor<*xf64>) -> tensor<*xf64>
    etoy.return
  }
}