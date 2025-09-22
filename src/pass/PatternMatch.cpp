//===- PatternMatch.cpp - TToy High Level Optimizer
//--------------------------===//
//
//
// This file implements a set of simple combiners for optimizing operations in
// the TToy dialect.
//
//===----------------------------------------------------------------------===//

#include "ttoy/Dialect.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

using namespace mlir;
using namespace ttoy;

namespace {
#include "generated/PatternMatch.inc"
} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
    results.add<TransposeTransposeOptPattern>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                FoldConstantReshapeOptPattern>(context);
}