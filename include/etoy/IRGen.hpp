//===- IRGen.td - Etoy dialect MLIRGen declarations ----------*- tablegen
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Generate Etoy-dialect from AST
//
//===----------------------------------------------------------------------===//

#ifndef ETOY_IRGEN_H
#define ETOY_IRGEN_H

#include <mlir/IR/MLIRContext.h>

namespace mlir {

template <typename OpTy> class OwningOpRef;
class ModuleOp;

} // namespace mlir

namespace etoy {
class ModuleAST;

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context,
                                          ModuleAST& module_ast);
} // namespace etoy

#endif // ETOY_IRGEN_H