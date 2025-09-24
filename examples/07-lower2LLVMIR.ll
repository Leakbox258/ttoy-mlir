; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"

declare !dbg !3 void @free(ptr)

declare !dbg !6 i32 @printf(ptr, ...)

declare !dbg !7 ptr @malloc(i64)

define void @main() !dbg !8 {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64)), !dbg !10
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0, !dbg !10
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1, !dbg !10
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2, !dbg !10
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 3, 3, 0, !dbg !10
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 2, 3, 1, !dbg !10
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 2, 4, 0, !dbg !10
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1, !dbg !10
  %9 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64)), !dbg !10
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %9, 0, !dbg !10
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, ptr %9, 1, !dbg !10
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 0, 2, !dbg !10
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 2, 3, 0, !dbg !10
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 3, 3, 1, !dbg !10
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 3, 4, 0, !dbg !10
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 1, 4, 1, !dbg !10
  %17 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64)), !dbg !11
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %17, 0, !dbg !11
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %17, 1, !dbg !11
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 0, 2, !dbg !11
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 3, 3, 0, !dbg !11
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 2, 3, 1, !dbg !11
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 2, 4, 0, !dbg !11
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 1, 4, 1, !dbg !11
  %25 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64)), !dbg !11
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %25, 0, !dbg !11
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %25, 1, !dbg !11
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 0, 2, !dbg !11
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 3, 3, 0, !dbg !11
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 2, 3, 1, !dbg !11
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 2, 4, 0, !dbg !11
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 1, 4, 1, !dbg !11
  %33 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64)), !dbg !12
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %33, 0, !dbg !12
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %33, 1, !dbg !12
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 0, 2, !dbg !12
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 2, 3, 0, !dbg !12
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 3, 3, 1, !dbg !12
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 3, 4, 0, !dbg !12
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 1, 4, 1, !dbg !12
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !12
  %42 = getelementptr double, ptr %41, i64 0, !dbg !12
  store double 1.000000e+00, ptr %42, align 8, !dbg !12
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !12
  %44 = getelementptr double, ptr %43, i64 1, !dbg !12
  store double 2.000000e+00, ptr %44, align 8, !dbg !12
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !12
  %46 = getelementptr double, ptr %45, i64 2, !dbg !12
  store double 3.000000e+00, ptr %46, align 8, !dbg !12
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !12
  %48 = getelementptr double, ptr %47, i64 3, !dbg !12
  store double 4.000000e+00, ptr %48, align 8, !dbg !12
  %49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !12
  %50 = getelementptr double, ptr %49, i64 4, !dbg !12
  store double 5.000000e+00, ptr %50, align 8, !dbg !12
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !12
  %52 = getelementptr double, ptr %51, i64 5, !dbg !12
  store double 6.000000e+00, ptr %52, align 8, !dbg !12
  br label %53, !dbg !11

53:                                               ; preds = %71, %0
  %54 = phi i64 [ 0, %0 ], [ %72, %71 ], !dbg !11
  %55 = icmp slt i64 %54, 3, !dbg !11
  br i1 %55, label %56, label %73, !dbg !11

56:                                               ; preds = %53
  br label %57, !dbg !11

57:                                               ; preds = %60, %56
  %58 = phi i64 [ 0, %56 ], [ %70, %60 ], !dbg !11
  %59 = icmp slt i64 %58, 2, !dbg !11
  br i1 %59, label %60, label %71, !dbg !11

60:                                               ; preds = %57
  %61 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1, !dbg !11
  %62 = mul i64 %58, 3, !dbg !11
  %63 = add i64 %62, %54, !dbg !11
  %64 = getelementptr double, ptr %61, i64 %63, !dbg !11
  %65 = load double, ptr %64, align 8, !dbg !11
  %66 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1, !dbg !11
  %67 = mul i64 %54, 2, !dbg !11
  %68 = add i64 %67, %58, !dbg !11
  %69 = getelementptr double, ptr %66, i64 %68, !dbg !11
  store double %65, ptr %69, align 8, !dbg !11
  %70 = add i64 %58, 1, !dbg !11
  br label %57, !dbg !11

71:                                               ; preds = %57
  %72 = add i64 %54, 1, !dbg !11
  br label %53, !dbg !11

73:                                               ; preds = %53
  br label %74, !dbg !11

74:                                               ; preds = %93, %73
  %75 = phi i64 [ 0, %73 ], [ %94, %93 ], !dbg !11
  %76 = icmp slt i64 %75, 3, !dbg !11
  br i1 %76, label %77, label %95, !dbg !11

77:                                               ; preds = %74
  br label %78, !dbg !11

78:                                               ; preds = %81, %77
  %79 = phi i64 [ 0, %77 ], [ %92, %81 ], !dbg !11
  %80 = icmp slt i64 %79, 2, !dbg !11
  br i1 %80, label %81, label %93, !dbg !11

81:                                               ; preds = %78
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1, !dbg !11
  %83 = mul i64 %75, 2, !dbg !11
  %84 = add i64 %83, %79, !dbg !11
  %85 = getelementptr double, ptr %82, i64 %84, !dbg !11
  %86 = load double, ptr %85, align 8, !dbg !11
  %87 = fmul double %86, %86, !dbg !11
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1, !dbg !11
  %89 = mul i64 %75, 2, !dbg !11
  %90 = add i64 %89, %79, !dbg !11
  %91 = getelementptr double, ptr %88, i64 %90, !dbg !11
  store double %87, ptr %91, align 8, !dbg !11
  %92 = add i64 %79, 1, !dbg !11
  br label %78, !dbg !11

93:                                               ; preds = %78
  %94 = add i64 %75, 1, !dbg !11
  br label %74, !dbg !11

95:                                               ; preds = %74
  br label %96, !dbg !10

96:                                               ; preds = %114, %95
  %97 = phi i64 [ 0, %95 ], [ %115, %114 ], !dbg !10
  %98 = icmp slt i64 %97, 2, !dbg !10
  br i1 %98, label %99, label %116, !dbg !10

99:                                               ; preds = %96
  br label %100, !dbg !10

100:                                              ; preds = %103, %99
  %101 = phi i64 [ 0, %99 ], [ %113, %103 ], !dbg !10
  %102 = icmp slt i64 %101, 3, !dbg !10
  br i1 %102, label %103, label %114, !dbg !10

103:                                              ; preds = %100
  %104 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1, !dbg !10
  %105 = mul i64 %101, 2, !dbg !10
  %106 = add i64 %105, %97, !dbg !10
  %107 = getelementptr double, ptr %104, i64 %106, !dbg !10
  %108 = load double, ptr %107, align 8, !dbg !10
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1, !dbg !10
  %110 = mul i64 %97, 3, !dbg !10
  %111 = add i64 %110, %101, !dbg !10
  %112 = getelementptr double, ptr %109, i64 %111, !dbg !10
  store double %108, ptr %112, align 8, !dbg !10
  %113 = add i64 %101, 1, !dbg !10
  br label %100, !dbg !10

114:                                              ; preds = %100
  %115 = add i64 %97, 1, !dbg !10
  br label %96, !dbg !10

116:                                              ; preds = %96
  br label %117, !dbg !10

117:                                              ; preds = %141, %116
  %118 = phi i64 [ 0, %116 ], [ %142, %141 ], !dbg !10
  %119 = icmp slt i64 %118, 3, !dbg !10
  br i1 %119, label %120, label %143, !dbg !10

120:                                              ; preds = %117
  br label %121, !dbg !10

121:                                              ; preds = %124, %120
  %122 = phi i64 [ 0, %120 ], [ %140, %124 ], !dbg !10
  %123 = icmp slt i64 %122, 2, !dbg !10
  br i1 %123, label %124, label %141, !dbg !10

124:                                              ; preds = %121
  %125 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1, !dbg !10
  %126 = mul i64 %118, 2, !dbg !10
  %127 = add i64 %126, %122, !dbg !10
  %128 = getelementptr double, ptr %125, i64 %127, !dbg !10
  %129 = load double, ptr %128, align 8, !dbg !10
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1, !dbg !10
  %131 = mul i64 %118, 3, !dbg !10
  %132 = add i64 %131, %122, !dbg !10
  %133 = getelementptr double, ptr %130, i64 %132, !dbg !10
  %134 = load double, ptr %133, align 8, !dbg !10
  %135 = fmul double %129, %134, !dbg !10
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1, !dbg !10
  %137 = mul i64 %118, 2, !dbg !10
  %138 = add i64 %137, %122, !dbg !10
  %139 = getelementptr double, ptr %136, i64 %138, !dbg !10
  store double %135, ptr %139, align 8, !dbg !10
  %140 = add i64 %122, 1, !dbg !10
  br label %121, !dbg !10

141:                                              ; preds = %121
  %142 = add i64 %118, 1, !dbg !10
  br label %117, !dbg !10

143:                                              ; preds = %117
  br label %144, !dbg !13

144:                                              ; preds = %159, %143
  %145 = phi i64 [ 0, %143 ], [ %161, %159 ], !dbg !13
  %146 = icmp slt i64 %145, 3, !dbg !13
  br i1 %146, label %147, label %162, !dbg !13

147:                                              ; preds = %144
  br label %148, !dbg !13

148:                                              ; preds = %151, %147
  %149 = phi i64 [ 0, %147 ], [ %158, %151 ], !dbg !13
  %150 = icmp slt i64 %149, 2, !dbg !13
  br i1 %150, label %151, label %159, !dbg !13

151:                                              ; preds = %148
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1, !dbg !13
  %153 = mul i64 %145, 2, !dbg !13
  %154 = add i64 %153, %149, !dbg !13
  %155 = getelementptr double, ptr %152, i64 %154, !dbg !13
  %156 = load double, ptr %155, align 8, !dbg !13
  %157 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %156), !dbg !13
  %158 = add i64 %149, 1, !dbg !13
  br label %148, !dbg !13

159:                                              ; preds = %148
  %160 = call i32 (ptr, ...) @printf(ptr @nl), !dbg !13
  %161 = add i64 %145, 1, !dbg !13
  br label %144, !dbg !13

162:                                              ; preds = %144
  %163 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 0, !dbg !12
  call void @free(ptr %163), !dbg !12
  %164 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 0, !dbg !11
  call void @free(ptr %164), !dbg !11
  %165 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0, !dbg !11
  call void @free(ptr %165), !dbg !11
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0, !dbg !10
  call void @free(ptr %166), !dbg !10
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0, !dbg !10
  call void @free(ptr %167), !dbg !10
  ret void, !dbg !14
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DISubprogram(name: "free", linkageName: "free", scope: !2, file: !2, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{}
!6 = !DISubprogram(name: "printf", linkageName: "printf", scope: !2, file: !2, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
!7 = !DISubprogram(name: "malloc", linkageName: "malloc", scope: !2, file: !2, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
!8 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !9, file: !9, line: 5, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!9 = !DIFile(filename: "07-lower2LLVMIR.ttoy", directory: "../examples")
!10 = !DILocation(line: 11, column: 13, scope: !8)
!11 = !DILocation(line: 8, column: 13, scope: !8)
!12 = !DILocation(line: 6, column: 13, scope: !8)
!13 = !DILocation(line: 12, column: 5, scope: !8)
!14 = !DILocation(line: 13, column: 5, scope: !8)