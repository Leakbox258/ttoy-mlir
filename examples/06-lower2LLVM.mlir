module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(6 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %6, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %7, %19[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %7, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %8, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.mlir.constant(3 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(6 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %23, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %24, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %24, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %25, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(3 : index) : i64
    %41 = llvm.mlir.constant(2 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.mlir.constant(6 : index) : i64
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = llvm.getelementptr %44[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %40, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %41, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %41, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %42, %55[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mlir.constant(3 : index) : i64
    %58 = llvm.mlir.constant(2 : index) : i64
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.mlir.constant(6 : index) : i64
    %61 = llvm.mlir.zero : !llvm.ptr
    %62 = llvm.getelementptr %61[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.call @malloc(%63) : (i64) -> !llvm.ptr
    %65 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.insertvalue %64, %65[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %57, %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %58, %70[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %58, %71[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %59, %72[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mlir.constant(2 : index) : i64
    %75 = llvm.mlir.constant(3 : index) : i64
    %76 = llvm.mlir.constant(1 : index) : i64
    %77 = llvm.mlir.constant(6 : index) : i64
    %78 = llvm.mlir.zero : !llvm.ptr
    %79 = llvm.getelementptr %78[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
    %81 = llvm.call @malloc(%80) : (i64) -> !llvm.ptr
    %82 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %83 = llvm.insertvalue %81, %82[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mlir.constant(0 : index) : i64
    %86 = llvm.insertvalue %85, %84[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %74, %86[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.insertvalue %75, %87[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.insertvalue %75, %88[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.insertvalue %76, %89[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.mlir.constant(0 : index) : i64
    %92 = llvm.mlir.constant(0 : index) : i64
    %93 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mlir.constant(3 : index) : i64
    %95 = llvm.mul %91, %94 : i64
    %96 = llvm.add %95, %92 : i64
    %97 = llvm.getelementptr %93[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %5, %97 : f64, !llvm.ptr
    %98 = llvm.mlir.constant(0 : index) : i64
    %99 = llvm.mlir.constant(1 : index) : i64
    %100 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(3 : index) : i64
    %102 = llvm.mul %98, %101 : i64
    %103 = llvm.add %102, %99 : i64
    %104 = llvm.getelementptr %100[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %4, %104 : f64, !llvm.ptr
    %105 = llvm.mlir.constant(0 : index) : i64
    %106 = llvm.mlir.constant(2 : index) : i64
    %107 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(3 : index) : i64
    %109 = llvm.mul %105, %108 : i64
    %110 = llvm.add %109, %106 : i64
    %111 = llvm.getelementptr %107[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %3, %111 : f64, !llvm.ptr
    %112 = llvm.mlir.constant(1 : index) : i64
    %113 = llvm.mlir.constant(0 : index) : i64
    %114 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.mlir.constant(3 : index) : i64
    %116 = llvm.mul %112, %115 : i64
    %117 = llvm.add %116, %113 : i64
    %118 = llvm.getelementptr %114[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %2, %118 : f64, !llvm.ptr
    %119 = llvm.mlir.constant(1 : index) : i64
    %120 = llvm.mlir.constant(1 : index) : i64
    %121 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.mlir.constant(3 : index) : i64
    %123 = llvm.mul %119, %122 : i64
    %124 = llvm.add %123, %120 : i64
    %125 = llvm.getelementptr %121[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %125 : f64, !llvm.ptr
    %126 = llvm.mlir.constant(1 : index) : i64
    %127 = llvm.mlir.constant(2 : index) : i64
    %128 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.mlir.constant(3 : index) : i64
    %130 = llvm.mul %126, %129 : i64
    %131 = llvm.add %130, %127 : i64
    %132 = llvm.getelementptr %128[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %0, %132 : f64, !llvm.ptr
    %133 = llvm.mlir.constant(0 : index) : i64
    %134 = llvm.mlir.constant(3 : index) : i64
    %135 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%133 : i64)
  ^bb1(%136: i64):  // 2 preds: ^bb0, ^bb5
    %137 = llvm.icmp "slt" %136, %134 : i64
    llvm.cond_br %137, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %138 = llvm.mlir.constant(0 : index) : i64
    %139 = llvm.mlir.constant(2 : index) : i64
    %140 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%138 : i64)
  ^bb3(%141: i64):  // 2 preds: ^bb2, ^bb4
    %142 = llvm.icmp "slt" %141, %139 : i64
    llvm.cond_br %142, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %143 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.mlir.constant(3 : index) : i64
    %145 = llvm.mul %141, %144 : i64
    %146 = llvm.add %145, %136 : i64
    %147 = llvm.getelementptr %143[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %148 = llvm.load %147 : !llvm.ptr -> f64
    %149 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.mlir.constant(2 : index) : i64
    %151 = llvm.mul %136, %150 : i64
    %152 = llvm.add %151, %141 : i64
    %153 = llvm.getelementptr %149[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %148, %153 : f64, !llvm.ptr
    %154 = llvm.add %141, %140 : i64
    llvm.br ^bb3(%154 : i64)
  ^bb5:  // pred: ^bb3
    %155 = llvm.add %136, %135 : i64
    llvm.br ^bb1(%155 : i64)
  ^bb6:  // pred: ^bb1
    %156 = llvm.mlir.constant(0 : index) : i64
    %157 = llvm.mlir.constant(3 : index) : i64
    %158 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%156 : i64)
  ^bb7(%159: i64):  // 2 preds: ^bb6, ^bb11
    %160 = llvm.icmp "slt" %159, %157 : i64
    llvm.cond_br %160, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %161 = llvm.mlir.constant(0 : index) : i64
    %162 = llvm.mlir.constant(2 : index) : i64
    %163 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb9(%161 : i64)
  ^bb9(%164: i64):  // 2 preds: ^bb8, ^bb10
    %165 = llvm.icmp "slt" %164, %162 : i64
    llvm.cond_br %165, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %166 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.mlir.constant(2 : index) : i64
    %168 = llvm.mul %159, %167 : i64
    %169 = llvm.add %168, %164 : i64
    %170 = llvm.getelementptr %166[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %171 = llvm.load %170 : !llvm.ptr -> f64
    %172 = llvm.fmul %171, %171 : f64
    %173 = llvm.extractvalue %56[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %174 = llvm.mlir.constant(2 : index) : i64
    %175 = llvm.mul %159, %174 : i64
    %176 = llvm.add %175, %164 : i64
    %177 = llvm.getelementptr %173[%176] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %172, %177 : f64, !llvm.ptr
    %178 = llvm.add %164, %163 : i64
    llvm.br ^bb9(%178 : i64)
  ^bb11:  // pred: ^bb9
    %179 = llvm.add %159, %158 : i64
    llvm.br ^bb7(%179 : i64)
  ^bb12:  // pred: ^bb7
    %180 = llvm.mlir.constant(0 : index) : i64
    %181 = llvm.mlir.constant(2 : index) : i64
    %182 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%180 : i64)
  ^bb13(%183: i64):  // 2 preds: ^bb12, ^bb17
    %184 = llvm.icmp "slt" %183, %181 : i64
    llvm.cond_br %184, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %185 = llvm.mlir.constant(0 : index) : i64
    %186 = llvm.mlir.constant(3 : index) : i64
    %187 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%185 : i64)
  ^bb15(%188: i64):  // 2 preds: ^bb14, ^bb16
    %189 = llvm.icmp "slt" %188, %186 : i64
    llvm.cond_br %189, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %190 = llvm.extractvalue %56[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.mlir.constant(2 : index) : i64
    %192 = llvm.mul %188, %191 : i64
    %193 = llvm.add %192, %183 : i64
    %194 = llvm.getelementptr %190[%193] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %195 = llvm.load %194 : !llvm.ptr -> f64
    %196 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %197 = llvm.mlir.constant(3 : index) : i64
    %198 = llvm.mul %183, %197 : i64
    %199 = llvm.add %198, %188 : i64
    %200 = llvm.getelementptr %196[%199] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %195, %200 : f64, !llvm.ptr
    %201 = llvm.add %188, %187 : i64
    llvm.br ^bb15(%201 : i64)
  ^bb17:  // pred: ^bb15
    %202 = llvm.add %183, %182 : i64
    llvm.br ^bb13(%202 : i64)
  ^bb18:  // pred: ^bb13
    %203 = llvm.mlir.constant(0 : index) : i64
    %204 = llvm.mlir.constant(3 : index) : i64
    %205 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb19(%203 : i64)
  ^bb19(%206: i64):  // 2 preds: ^bb18, ^bb23
    %207 = llvm.icmp "slt" %206, %204 : i64
    llvm.cond_br %207, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    %208 = llvm.mlir.constant(0 : index) : i64
    %209 = llvm.mlir.constant(2 : index) : i64
    %210 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb21(%208 : i64)
  ^bb21(%211: i64):  // 2 preds: ^bb20, ^bb22
    %212 = llvm.icmp "slt" %211, %209 : i64
    llvm.cond_br %212, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %213 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.mlir.constant(2 : index) : i64
    %215 = llvm.mul %206, %214 : i64
    %216 = llvm.add %215, %211 : i64
    %217 = llvm.getelementptr %213[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %218 = llvm.load %217 : !llvm.ptr -> f64
    %219 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.mlir.constant(3 : index) : i64
    %221 = llvm.mul %206, %220 : i64
    %222 = llvm.add %221, %211 : i64
    %223 = llvm.getelementptr %219[%222] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %224 = llvm.load %223 : !llvm.ptr -> f64
    %225 = llvm.fmul %218, %224 : f64
    %226 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.mlir.constant(2 : index) : i64
    %228 = llvm.mul %206, %227 : i64
    %229 = llvm.add %228, %211 : i64
    %230 = llvm.getelementptr %226[%229] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %225, %230 : f64, !llvm.ptr
    %231 = llvm.add %211, %210 : i64
    llvm.br ^bb21(%231 : i64)
  ^bb23:  // pred: ^bb21
    %232 = llvm.add %206, %205 : i64
    llvm.br ^bb19(%232 : i64)
  ^bb24:  // pred: ^bb19
    %233 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %234 = llvm.mlir.constant(0 : index) : i64
    %235 = llvm.getelementptr %233[%234, %234] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %236 = llvm.mlir.addressof @nl : !llvm.ptr
    %237 = llvm.mlir.constant(0 : index) : i64
    %238 = llvm.getelementptr %236[%237, %237] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %239 = llvm.mlir.constant(0 : index) : i64
    %240 = llvm.mlir.constant(3 : index) : i64
    %241 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb25(%239 : i64)
  ^bb25(%242: i64):  // 2 preds: ^bb24, ^bb29
    %243 = llvm.icmp "slt" %242, %240 : i64
    llvm.cond_br %243, ^bb26, ^bb30
  ^bb26:  // pred: ^bb25
    %244 = llvm.mlir.constant(0 : index) : i64
    %245 = llvm.mlir.constant(2 : index) : i64
    %246 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb27(%244 : i64)
  ^bb27(%247: i64):  // 2 preds: ^bb26, ^bb28
    %248 = llvm.icmp "slt" %247, %245 : i64
    llvm.cond_br %248, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %249 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.mlir.constant(2 : index) : i64
    %251 = llvm.mul %242, %250 : i64
    %252 = llvm.add %251, %247 : i64
    %253 = llvm.getelementptr %249[%252] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %254 = llvm.load %253 : !llvm.ptr -> f64
    %255 = llvm.call @printf(%235, %254) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %256 = llvm.add %247, %246 : i64
    llvm.br ^bb27(%256 : i64)
  ^bb29:  // pred: ^bb27
    %257 = llvm.call @printf(%238) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %258 = llvm.add %242, %241 : i64
    llvm.br ^bb25(%258 : i64)
  ^bb30:  // pred: ^bb25
    %259 = llvm.extractvalue %90[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%259) : (!llvm.ptr) -> ()
    %260 = llvm.extractvalue %73[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%260) : (!llvm.ptr) -> ()
    %261 = llvm.extractvalue %56[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%261) : (!llvm.ptr) -> ()
    %262 = llvm.extractvalue %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%262) : (!llvm.ptr) -> ()
    %263 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%263) : (!llvm.ptr) -> ()
    llvm.return
  }
}