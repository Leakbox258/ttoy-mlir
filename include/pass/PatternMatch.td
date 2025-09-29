//===- PatternMatch.td - Etoy dialect PatternMatch passes ----------*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Defines the PatternMatch passes of Etoy dialect using DRR
//
//===----------------------------------------------------------------------===//

#ifndef ETOY_DRR_PATTERN_MATCH
#define ETOY_DRR_PATTERN_MATCH

include "mlir/IR/PatternBase.td"
include "etoy/Ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
//     list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

// reduntant reshape eli
// Pattern will only sourcePattern and resultPatterns
// ADDME: fix point
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

// reduntant transpose eli
// ADDME: fix point
def TransposeTransposeOptPattern : Pat<(TransposeOp (TransposeOp $x)), (replaceWithValue $x)>;

// ReshapeConstant convert
// Pattern Match using NativeCodeCall impled by c++
def ReshapeConstant : NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))
>;

// Reshape with same input and output type
// Pattern use condition Constraint
def IsTypesIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
    (ReshapeOp:$res $arg),
    (replaceWithValue $arg),
    [(IsTypesIdentical $arg, $res)]
>;

#endif //