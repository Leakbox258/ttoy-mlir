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
#include "generated/Dialect.cpp.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/LogicalResult.h"

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

using namespace mlir;
using namespace mlir::ttoy;

/// where to registrate the types and ops on current context
void TToyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "generated/Ops.cpp.inc"
        >();
}

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser& parser,
                                       mlir::OperationState& result) {
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operands_loc = parser.getCurrentLocation();
    Type type;

    // 1. get preparations
    // parse<>s return true when ParseResult::failed() return true
    if (parser.parseOperandList(operands, 2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type)) {
        return mlir::failure();
    }

    // 2. check if is Function Type, if so, handle inputs and output carefully
    if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operands_loc,
                                   result.operands)) {
            return mlir::failure();
        } else {
            result.addTypes(funcType.getResults());
            return mlir::success();
        }
    } else {
        if (parser.resolveOperands(operands, type, result.operands)) {
            return mlir::failure();
        } else {
            result.addTypes(type);
            return mlir::success();
        }
    }
}

static void printBinaryOp(mlir::OpAsmPrinter& printer, mlir::Operation* op) {

    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // here we haven't consider tensor broadcast issues
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    } else {
        // trate as function
        printer.printFunctionalType(op->getOperandTypes(),
                                    op->getResultTypes());
    }
}

void ConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                       double value) {
    auto dataType = RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);

    ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser& parser,
                                    mlir::OperationState& result) {
    mlir::DenseElementsAttr value;

    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes)) {
        return failure();
    } else {
        result.addTypes(value.getType());
        return success();
    }
}

void ConstantOp::print(mlir::OpAsmPrinter& printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
    printer << getValue();
}

// verify if resultType of this operation eq to the attrType
llvm::LogicalResult ConstantOp::verify() {
    auto resultType =
        llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());

    if (!resultType) {
        return success();
    }

    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());

    if (attrType.getRank() != resultType.getRank()) {
        return emitOpError(
                   "return type must match the one of the attached value "
                   "attribute: ")
               << attrType.getRank() << " != " << resultType.getRank();
    }

    // check each dimension
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return emitOpError("return type shape mismatches its attribute at "
                               "dimension ")
                   << dim << ": " << attrType.getShape()[dim]
                   << " != " << resultType.getShape()[dim];
        }
    }

    return mlir::success();
}

// AddOp
void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
    return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter& printer) {
    printBinaryOp(printer, *this);
}

// MulOp
void MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
    return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter& printer) {
    printBinaryOp(printer, *this);
}

// FuncOp
void FuncOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
    auto buildFuncType =
        [](mlir::Builder& builder, llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string&) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter& printer) {
    mlir::function_interface_impl::printFunctionOp(
        printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
        getResAttrsAttrName());
}

// CallOp
void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   StringRef callee, ArrayRef<mlir::Value> arguments) {

    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee",
                       mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

// ReturnOp
llvm::LogicalResult ReturnOp::verify() {
    auto function = cast<FuncOp>((*this)->getParentOp());

    if (getNumOperands() > 1) {
        return emitOpError() << "execpts at most 1 return operand";
    }

    const auto& results = function.getFunctionType().getResults();

    if (getNumOperands() != results.size())
        return emitOpError()
               << "does not return the same number of values ("
               << getNumOperands() << ") as the enclosing function ("
               << results.size() << ")";

    if (!hasOperand()) {
        return mlir::success();
    }

    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    // type checks
    if (inputType == resultType ||
        llvm::isa<mlir::UnrankedTensorType>(inputType) ||
        llvm::isa<mlir::UnrankedTensorType>(resultType)) {
        return mlir::success();
    }

    return emitError() << "type of return operand (" << inputType
                       << ") doesn't match function result type (" << resultType
                       << ")";
}

// TransposeOp
void TransposeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        mlir::Value value) {

    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(value);
}

llvm::LogicalResult TransposeOp::verify() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
    auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());

    if (!inputType || !resultType) {
        return mlir::success();
    }

    auto inputShape = inputType.getShape();

    if (!std::equal(inputShape.begin(), inputShape.end(),
                    resultType.getShape().rbegin())) {
        return emitError()
               << "expected result shape to be a transpose of the input";
    }

    return mlir::success();
}

// additional definitions

#define GET_OP_CLASSES
#include "generated/Ops.cpp.inc"