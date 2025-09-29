//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef ETOY_SHAPEINFERENCEINTERFACE_HPP_
#define ETOY_SHAPEINFERENCEINTERFACE_HPP_

#include <mlir/IR/OpDefinition.h>

namespace mlir {
namespace etoy {

#include "generated/ShapeInferenceOpInterface.h.inc"

}
} // namespace mlir

#endif