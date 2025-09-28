//===- IRGen.td - TToy dialect MLIRGen declarations ----------*- tablegen
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Generate TToy-dialect from AST
//
//===----------------------------------------------------------------------===//

#ifndef TTOY_IRGEN_H
#define TTOY_IRGEN_H

#include <mlir/IR/MLIRContext.h>

namespace mlir {

template <typename OpTy> class OwningOpRef;
class ModuleOp;

} // namespace mlir

namespace ttoy {
class ModuleAST;

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context,
                                          ModuleAST& module_ast);
} // namespace ttoy

#endif // TTOY_IRGEN_H