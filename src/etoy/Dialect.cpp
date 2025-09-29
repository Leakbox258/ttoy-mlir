//===- Dialect.td - Etoy dialect definitions ----------*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Defines the Etoy dialect.
// include Dialect itself, Ops, Builtins, all in one
//
//===----------------------------------------------------------------------===//

#include "etoy/Dialect.hpp"
#include "generated/Dialect.cpp.inc"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>
#include <string>

using namespace mlir;
using namespace mlir::etoy;

/// etoy dialect inliner

struct EtoyInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    // Analysis Hooks

    /// all call to no-builtin functions will inline
    bool isLegalToInline(Operation* call, Operation* callable,
                         bool would_be_cloned) const final {
        return true;
    }

    /// all operations can be inlined
    bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
        return true;
    }

    /// all function will inline
    bool isLegalToInline(Region*, Region*, bool, IRMapping&) const final {
        return true;
    }

    // Transformation Hooks

    void handleTerminator(Operation* op,
                          ValueRange values_to_repl) const final {
        auto return_op = cast<ReturnOp>(op);

        assert(return_op->getNumOperands() == values_to_repl.size());
        for (const auto& it : llvm::enumerate(return_op->getOperands())) {
            values_to_repl[it.index()].replaceAllUsesWith(it.value());
        }
    }

    Operation* materializeCallConversion(OpBuilder& builder, Value input,
                                         Type resultType,
                                         Location conversionLoc) const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

/// where to registrate the types and ops on current context
void EtoyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "generated/Ops.cpp.inc"
        >();
    addInterfaces<EtoyInlinerInterface>();
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

void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

// SubOp
void SubOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult SubOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
    return parseBinaryOp(parser, result);
}

void SubOp::print(mlir::OpAsmPrinter& printer) {
    printBinaryOp(printer, *this);
}

void SubOp::inferShapes() { getResult().setType(getLhs().getType()); }

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

void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

// DivOp
void DivOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult DivOp::parse(mlir::OpAsmParser& parser,
                               mlir::OperationState& result) {
    return parseBinaryOp(parser, result);
}

void DivOp::print(mlir::OpAsmPrinter& printer) {
    printBinaryOp(printer, *this);
}

void DivOp::inferShapes() { getResult().setType(getLhs().getType()); }

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

CallInterfaceCallable CallOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
    (*this)->setAttr("callee", llvm::cast<SymbolRefAttr>(callee));
}

Operation::operand_range CallOp::getArgOperands() { return getInputs(); }

// ScanOp
llvm::LogicalResult ScanOp::verify() {
    auto input_type = llvm::dyn_cast<RankedTensorType>(
        getOperand().getType()); // sym_name type

    auto template_type = llvm::dyn_cast<RankedTensorType>(getTemplateType());

    if (!input_type || !template_type) {
        return mlir::success();
    }

    auto input_shape = input_type.getShape();

    if (!std::equal(input_shape.begin(), input_shape.end(),
                    template_type.getShape().begin())) {

        return emitError()
               << "expected input shape to be equal with template shape";
    }

    return mlir::success();
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

MutableOperandRange CallOp::getArgOperandsMutable() {
    return getInputsMutable();
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

void TransposeOp::inferShapes() {
    auto array_ty = llvm::cast<RankedTensorType>(getOperand().getType());

    llvm::SmallVector<int64_t, 2> dims(llvm::reverse(array_ty.getShape()));

    getResult().setType(RankedTensorType::get(dims, array_ty.getElementType()));
}

// castop
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;

    TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
    TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
    if (!input || !output || input.getElementType() != output.getElementType())
        return false;

    return !input.hasRank() || !output.hasRank() || input == output;
}

// dotop
void DotOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) {

    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(lhs);
    state.addOperands(rhs);
}

void DotOp::inferShapes() {
    getResult().setType(
        RankedTensorType::get({1}, getLhs().getType().getElementType()));
}

llvm::LogicalResult DotOp::verify() {
    const auto lhs_shape = getLhs().getType().getShape();
    const auto rhs_shape = getRhs().getType().getShape();

    // no broadcasting
    if (lhs_shape != rhs_shape) {
        return mlir::failure();
    }

    // all 1D
    if (lhs_shape.size() != 1 || rhs_shape.size() != 1) {
        return mlir::failure();
    }

    return mlir::success();
}

// mmop
void MMOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                 mlir::Value lhs, mlir::Value rhs) {

    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(lhs);
    state.addOperands(rhs);
}

void MMOp::inferShapes() {
    const auto lhs_shape = getLhs().getType().getShape();
    const auto rhs_shape = getRhs().getType().getShape();

    llvm::SmallVector<int64_t, 2> dims{lhs_shape[0], rhs_shape[1]};

    getResult().setType(
        RankedTensorType::get(dims, getLhs().getType().getElementType()));
}

llvm::LogicalResult MMOp::verify() {
    const auto lhs_shape = getLhs().getType().getShape();
    const auto rhs_shape = getRhs().getType().getShape();

    // all 2D
    if (lhs_shape.size() != 2 || rhs_shape.size() != 2) {
        return mlir::failure();
    }

    if (lhs_shape[1] != rhs_shape[0]) {
        return mlir::failure();
    }

    return mlir::success();
}

// bmmop
void BMMOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) {

    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(lhs);
    state.addOperands(rhs);
}

void BMMOp::inferShapes() {
    const auto lhs_shape = getLhs().getType().getShape();
    const auto rhs_shape = getRhs().getType().getShape();

    llvm::SmallVector<int64_t, 2> dims{lhs_shape[0], lhs_shape[1],
                                       rhs_shape[2]};

    getResult().setType(
        RankedTensorType::get(dims, getLhs().getType().getElementType()));
}

llvm::LogicalResult BMMOp::verify() {
    const auto lhs_shape = getLhs().getType().getShape();
    const auto rhs_shape = getRhs().getType().getShape();

    // all 3D
    if (lhs_shape.size() != 3 || rhs_shape.size() != 3) {
        return mlir::failure();
    }

    // batch size
    if (lhs_shape[0] != rhs_shape[0]) {
        return mlir::failure();
    }

    if (lhs_shape[2] != rhs_shape[1]) {
        return mlir::failure();
    }

    return mlir::success();
}

#define GET_OP_CLASSES
#include "generated/Ops.cpp.inc"