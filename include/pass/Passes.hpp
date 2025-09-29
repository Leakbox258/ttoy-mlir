//===- Passes.h - Etoy Passes Definition
//-----------------------------------===//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Etoy.
//
//===----------------------------------------------------------------------===//

#ifndef ETOY_PASSES_H
#define ETOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace etoy {
std::unique_ptr<Pass> createShapeInferencePass();

std::unique_ptr<mlir::Pass> createLowerToAffinePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace etoy
} // namespace mlir

#endif // ETOY_PASSES_H