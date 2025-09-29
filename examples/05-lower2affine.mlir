module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_5 = memref.alloc() : memref<2x3xf64>
    %alloc_6 = memref.alloc() : memref<3x2xf64>
    %alloc_7 = memref.alloc() : memref<3x2xf64>
    %alloc_8 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_4, %alloc_8[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_8[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_8[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_8[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_8[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_8[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_8[%arg1, %arg0] : memref<2x3xf64>
        affine.store %0, %alloc_7[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_7[%arg0, %arg1] : memref<3x2xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc_6[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 3 {
        %0 = affine.load %alloc_6[%arg1, %arg0] : memref<3x2xf64>
        affine.store %0, %alloc_5[%arg0, %arg1] : memref<2x3xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_7[%arg0, %arg1] : memref<3x2xf64>
        %1 = affine.load %alloc_5[%arg0, %arg1] : memref<2x3xf64>
        %2 = arith.mulf %0, %1 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    etoy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_8 : memref<2x3xf64>
    memref.dealloc %alloc_7 : memref<3x2xf64>
    memref.dealloc %alloc_6 : memref<3x2xf64>
    memref.dealloc %alloc_5 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}