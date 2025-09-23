//===- IRGen.td - TToy dialect MLIRGen definitions ----------*- tablegen
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Generate TToy-dialect from AST
//
//===----------------------------------------------------------------------===//

#include "ttoy/IRGen.hpp"
#include "mlir/IR/OwningOpRef.h"
#include "parser/AST.h"
#include "parser/Lexer.h"
#include "ttoy/Dialect.hpp"

#include <cstdio>
#include <functional>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/LogicalResult.h>
#include <numeric>
#include <vector>

using namespace ttoy;

namespace {
class MLIRGenImpl {
  private:
    mlir::ModuleOp ttoy_module;
    mlir::OpBuilder builder;
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    /// location when building from AST
    mlir::Location loc(const Location& loc) {
        return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
                                         loc.line, loc.col);
    }

    /// declartion of vars, including handling scopes
    llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (symbolTable.count(var)) {
            return mlir::failure();
        }
        symbolTable.insert(var, value);
        return mlir::success();
    }

    /// build types
    mlir::Type getType(const VarType& type) { return getType(type.shape); }
    mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
        // If the shape is empty, then this type is unranked.
        if (shape.empty())
            return mlir::UnrankedTensorType::get(builder.getF64Type());

        // Otherwise, we use the given shape.
        return mlir::RankedTensorType::get(shape, builder.getF64Type());
    }

    /// overloads of mlirGen
    mlir::ttoy::FuncOp mlirGen(PrototypeAST& proto) {
        auto location = loc(proto.loc());

        llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                                  getType(VarType{}));

        auto funcType = builder.getFunctionType(argTypes, std::nullopt);

        return builder.create<mlir::ttoy::FuncOp>(location, proto.getName(),
                                                  funcType);
    }

    mlir::ttoy::FuncOp mlirGen(FunctionAST& func_ast) {
        // set inline private (set entry)

        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(
            symbolTable);

        builder.setInsertionPointToEnd(ttoy_module.getBody());
        mlir::ttoy::FuncOp function = mlirGen(*func_ast.getProto());

        if (!function) {
            return nullptr;
        }

        // 1. building body
        mlir::Block& entry_block = function.front();
        auto proto_args = func_ast.getProto()->getArgs();

        // 2. declare args
        for (const auto nameValue :
             llvm::zip(proto_args, entry_block.getArguments())) {

            if (failed(declare(std::get<0>(nameValue)->getName(),
                               std::get<1>(nameValue)))) {
                return nullptr;
            }
        }

        builder.setInsertionPointToStart(&entry_block);

        // 3. emit body (exprs list)
        if (mlir::failed(mlirGen(*func_ast.getBody()))) {
            function.erase();
            return nullptr;
        }

        // 4. trait the last operation as return or simply add a return
        // FIXME: this will be tricky when include control flow or REPL into
        // this language
        mlir::ttoy::ReturnOp returnOp;
        if (!entry_block.empty()) {
            returnOp = dyn_cast<mlir::ttoy::ReturnOp>(entry_block.back());
        }
        if (!returnOp) {
            builder.create<mlir::ttoy::ReturnOp>(
                loc(func_ast.getProto()->loc()));
        } else if (returnOp.hasOperand()) {
            function.setType(builder.getFunctionType(
                function.getFunctionType().getInputs(), getType(VarType{})));
        }

        if (func_ast.getProto()->getName() != "main") {
            function.setPrivate();
        }

        return function;
    }

    mlir::Value mlirGen(BinaryExprAST& binary_ast) {
        auto lhs = mlirGen(*binary_ast.getLHS());
        if (!lhs) {
            return nullptr;
        }

        auto rhs = mlirGen(*binary_ast.getRHS());
        if (!rhs) {
            return nullptr;
        }

        auto location = loc(binary_ast.loc());

        switch (binary_ast.getOp()) {
        case '+':
            return builder.create<mlir::ttoy::AddOp>(location, lhs, rhs);
        case '*':
            return builder.create<mlir::ttoy::MulOp>(location, lhs, rhs);
        }

        mlir::emitError(location, "invalid binary operation '")
            << binary_ast.getOp() << "'";

        return nullptr;
    }

    llvm::LogicalResult mlirGen(ReturnExprAST& ret_ast) {
        auto location = loc(ret_ast.loc());

        mlir::Value expr = nullptr;

        if (ret_ast.getExpr().has_value()) {
            if (!(expr = mlirGen(**ret_ast.getExpr())))
                return mlir::failure();
        }

        builder.create<mlir::ttoy::ReturnOp>(
            location,
            expr ? llvm::ArrayRef(expr) : llvm::ArrayRef<mlir::Value>());

        return mlir::success();
    }

    mlir::Value mlirGen(VariableExprAST& expr_ast) {
        if (auto variable = symbolTable.lookup(expr_ast.getName())) {
            return variable;
        }

        auto location = loc(expr_ast.loc());

        mlir::emitError(location, "error: unknown variable'")
            << expr_ast.getName() << "'";

        return nullptr;
    }

    mlir::Value mlirGen(LiteralExprAST& lit_ast) {
        auto type = getType(lit_ast.getDims());
        auto location = loc(lit_ast.loc());

        // try flatten tensor, and avoid using smallvector
        std::vector<double> datas;

        datas.reserve(std::accumulate(lit_ast.getDims().begin(),
                                      lit_ast.getDims().end(), 1,
                                      std::multiplies<unsigned>()));

        collectData(lit_ast, datas);

        // build type attr
        mlir::Type elementType = builder.getF64Type();
        auto dataType =
            mlir::RankedTensorType::get(lit_ast.getDims(), elementType);

        auto dataAttribute =
            mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(datas));

        return builder.create<mlir::ttoy::ConstantOp>(location, type,
                                                      dataAttribute);
    }

    void collectData(ExprAST& expr, std::vector<double>& datas) {
        if (auto* lit = llvm::dyn_cast<LiteralExprAST>(&expr)) {
            for (auto& value : lit->getValues())
                collectData(*value, datas);
            return;
        }

        assert(llvm::isa<NumberExprAST>(expr) &&
               "expected literal or number expr");

        datas.push_back(llvm::cast<NumberExprAST>(expr).getValue());
    }

    /// need to handle builtins
    mlir::Value mlirGen(CallExprAST& call_ast) {
        llvm::StringRef callee = call_ast.getCallee();
        auto location = loc(call_ast.loc());

        llvm::SmallVector<mlir::Value, 4> operands;
        for (auto& expr : call_ast.getArgs()) {
            auto arg = mlirGen(*expr);
            if (!arg) {
                return nullptr;
            }
            operands.push_back(arg);
        }

        // recognize builtins
        if (callee == "transpose") {
            if (call_ast.getArgs().size() != 1) {
                emitError(location,
                          "MLIR codegen encountered an error: ttoy.transpose "
                          "does not accept multiple arguments");
                return nullptr;
            }
            return builder.create<mlir::ttoy::TransposeOp>(location,
                                                           operands[0]);
        }

        return builder.create<mlir::ttoy::CallOp>(location, callee, operands);
    }

    llvm::LogicalResult mlirGen(PrintExprAST& call) {
        auto arg = mlirGen(*call.getArg());
        if (!arg)
            return mlir::failure();

        builder.create<mlir::ttoy::PrintOp>(loc(call.loc()), arg);
        return mlir::success();
    }

    mlir::Value mlirGen(NumberExprAST& num) {
        return builder.create<mlir::ttoy::ConstantOp>(loc(num.loc()),
                                                      num.getValue());
    }

    /// Dispatch codegen for the right expression subclass using RTTI.
    mlir::Value mlirGen(ExprAST& expr) {
        switch (expr.getKind()) {
        case ExprAST::Expr_BinOp:
            return mlirGen(llvm::cast<BinaryExprAST>(expr));
        case ExprAST::Expr_Var:
            return mlirGen(llvm::cast<VariableExprAST>(expr));
        case ExprAST::Expr_Literal:
            return mlirGen(llvm::cast<LiteralExprAST>(expr));
        case ExprAST::Expr_Call:
            return mlirGen(llvm::cast<CallExprAST>(expr));
        case ExprAST::Expr_Num:
            return mlirGen(llvm::cast<NumberExprAST>(expr));
        default:
            emitError(loc(expr.loc()))
                << "MLIR codegen encountered an unhandled expr kind '"
                << llvm::Twine(expr.getKind()) << "'";
            return nullptr;
        }
    }

    mlir::Value mlirGen(VarDeclExprAST& vardecl) {
        auto* init = vardecl.getInitVal();
        if (!init) {
            emitError(loc(vardecl.loc()),
                      "missing initializer in variable declaration");
            return nullptr;
        }

        mlir::Value value = mlirGen(*init);
        if (!value)
            return nullptr;

        if (!vardecl.getType().shape.empty()) {
            value = builder.create<mlir::ttoy::ReshapeOp>(
                loc(vardecl.loc()), getType(vardecl.getType()), value);
        }

        // Register the value in the symbol table.
        if (failed(declare(vardecl.getName(), value)))
            return nullptr;
        return value;
    }

    llvm::LogicalResult mlirGen(ExprASTList& block_ast) {
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
            symbolTable);
        for (auto& expr : block_ast) {
            // Specific handling for variable declarations, return statement,
            // and print. These can only appear in block list and not in nested
            // expressions.
            if (auto* vardecl = llvm::dyn_cast<VarDeclExprAST>(expr.get())) {
                if (!mlirGen(*vardecl))
                    return mlir::failure();
                continue;
            }
            if (auto* ret = llvm::dyn_cast<ReturnExprAST>(expr.get()))
                return mlirGen(*ret);
            if (auto* print = llvm::dyn_cast<PrintExprAST>(expr.get())) {
                if (mlir::failed(mlirGen(*print)))
                    return mlir::success();
                continue;
            }

            // Generic expression dispatch codegen.
            if (!mlirGen(*expr))
                return mlir::failure();
        }
        return mlir::success();
    }

  public:
    MLIRGenImpl(mlir::MLIRContext& context) : builder(&context) {}

    mlir::ModuleOp mlirGen(ModuleAST& module_ast) {
        // We create an empty MLIR module and codegen functions one at a time
        // and add them to the module.
        ttoy_module = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (FunctionAST& f : module_ast) {
            mlirGen(f);
        }

        if (failed(mlir::verify(ttoy_module))) {
            ttoy_module.emitError("module verification error");
            return nullptr;
        }

        return ttoy_module;
    }
};
} // namespace

namespace ttoy {
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context,
                                          ModuleAST& module_ast) {
    return MLIRGenImpl(context).mlirGen(module_ast);
}
} // namespace ttoy