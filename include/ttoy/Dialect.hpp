//===- Dialect.td - TToy dialect declarations ----------*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Declarations the TToy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TTOY_DIALECT_HPP
#define TTOY_DIALECT_HPP

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// include decls from ShapeInferenceOpInferface.td
#include "pass/ShapeInferenceOpInterface.hpp"

// include from generate dir
#include "generated/Dialect.h.inc"

#define GET_OP_CLASSES
#include "generated/Ops.h.inc"

// TODO

#endif