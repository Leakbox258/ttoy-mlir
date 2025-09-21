//===- Dialect.td - TToy dialect definitions ----------*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Defines the TToy dialect.
// include Dialect itself, Ops, Builtins, all in one
//
//===----------------------------------------------------------------------===//

#include "ttoy/Dialect.hpp"

#include <algorithm>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <string>
