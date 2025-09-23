//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef TTOY_SHAPEINFERENCEINTERFACE_HPP_
#define TTOY_SHAPEINFERENCEINTERFACE_HPP_

#include <mlir/IR/OpDefinition.h>

namespace mlir {
namespace ttoy {

#include "generated/ShapeInferenceOpInterface.h.inc"

}
} // namespace mlir

#endif