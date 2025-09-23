//===- Passes.h - TToy Passes Definition
//-----------------------------------===//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for TToy.
//
//===----------------------------------------------------------------------===//

#ifndef TTOY_PASSES_H
#define TTOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace ttoy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace ttoy
} // namespace mlir

#endif // TTOY_PASSES_H