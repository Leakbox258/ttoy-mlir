//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'ttoy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "pass/Passes.hpp"
#include "ttoy/Dialect.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/Support/Casting.h>

#include <memory>
#include <optional>
#include <utility>

using namespace mlir;

namespace {
// ttoy.print -> printf with scf
class PrintOpLowering : public ConversionPattern {
  public:
    explicit PrintOpLowering(MLIRContext* context)
        : ConversionPattern(ttoy::PrintOp::getOperationName(), 1, context) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const override {
        auto* context = rewriter.getContext();

        auto memref_type = llvm::cast<MemRefType>(*op->operand_type_begin());
        auto memref_shape = memref_type.getShape();
        auto loc = op->getLoc();
        ModuleOp parent_module = op->getParentOfType<ModuleOp>();

        auto printf_ref = getOrInsertPrintf(rewriter, parent_module);
        Value format_specifier_cast = getOrCreateGlobalString(
            loc, rewriter, "frmt_spec", StringRef("%lf \0", 5), parent_module);
        Value newline_cast = getOrCreateGlobalString(
            loc, rewriter, "nl", StringRef("\n\0", 2), parent_module);

        llvm::SmallVector<Value, 4> loopIvs;
        for (unsigned i = 0, e = memref_shape.size(); i != e; ++i) {
            auto lower_bound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto upper_bound =
                rewriter.create<arith::ConstantIndexOp>(loc, memref_shape[i]);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            auto loop = rewriter.create<scf::ForOp>(loc, lower_bound,
                                                    upper_bound, step);

            for (Operation& nested : *loop.getBody()) {
                rewriter.eraseOp(&nested);
            }

            loopIvs.push_back(loop.getInductionVar());

            // add terminate for loop body
            rewriter.setInsertionPointToEnd(loop.getBody());

            if (i != e - 1) {
                rewriter.create<LLVM::CallOp>(loc, getPrintfType(context),
                                              printf_ref, newline_cast);
            }

            rewriter.create<scf::YieldOp>(loc);
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        // Generate a call to printf for the current element of the loop.
        auto print_op = cast<ttoy::PrintOp>(op);
        auto element_load =
            rewriter.create<memref::LoadOp>(loc, print_op.getInput(), loopIvs);

        rewriter.create<LLVM::CallOp>(
            loc, getPrintfType(context), printf_ref,
            ArrayRef<Value>({format_specifier_cast, element_load}));

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);
        return success();
    }

  private:
    /// Create a function declaration for printf, the signature is:
    ///   * `i32 (i8*, ...)`
    static LLVM::LLVMFunctionType getPrintfType(MLIRContext* context) {
        auto llvm_i32Ty = IntegerType::get(context, 32);
        auto llvm_ptrTy = LLVM::LLVMPointerType::get(context); // anomynous
        auto llvm_fnType = LLVM::LLVMFunctionType::get(llvm_i32Ty, llvm_ptrTy,
                                                       /*isVarArg=*/true);
        return llvm_fnType;
    }

    /// Return a symbol reference to the printf function, inserting it into the
    /// module if necessary.
    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter& rewriter,
                                               ModuleOp module) {
        auto* context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
            return SymbolRefAttr::get(context, "printf");
        }

        // Insert the printf function decl
        PatternRewriter::InsertionGuard insert_guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                          getPrintfType(context));
        return SymbolRefAttr::get(context, "printf");
    }

    /// Return a value representing an access into a global string with the
    /// given name, creating the string if necessary.
    static Value getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                         StringRef name, StringRef value,
                                         ModuleOp module) {
        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
            OpBuilder::InsertionGuard insert_guard(builder);

            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(
                loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
                builder.getStringAttr(value),
                /*alignment=*/0);
        }

        // gep
        Value global_ptr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                      builder.getIndexAttr(0));

        return builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(builder.getContext()),
            global.getType(), global_ptr, ArrayRef<Value>({cst0, cst0}));
    }
};

// ttoy.scan -> scanf with scf
class ScanOpLowering : public ConversionPattern {
  public:
    explicit ScanOpLowering(MLIRContext* context)
        : ConversionPattern(ttoy::ScanOp::getOperationName(), 1, context) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const override {
        auto* context = rewriter.getContext();

        auto memref_type = llvm::cast<MemRefType>(*op->operand_type_begin());

        auto memref_shape = memref_type.getShape();
        auto loc = op->getLoc();
        ModuleOp parent_module = op->getParentOfType<ModuleOp>();

        auto scanf_ref = getOrInsertScanf(rewriter, parent_module);
        Value format_specifier_cast = getOrCreateGlobalString(
            loc, rewriter, "frmt_spec", StringRef("%lf \0", 5), parent_module);

        Value newline_cast = getOrCreateGlobalString(
            loc, rewriter, "nl", StringRef("\n\0", 2), parent_module);

        llvm::SmallVector<Value, 4> loopIvs;
        for (unsigned i = 0, e = memref_shape.size(); i != e; ++i) {
            auto lower_bound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto upper_bound =
                rewriter.create<arith::ConstantIndexOp>(loc, memref_shape[i]);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            auto loop = rewriter.create<scf::ForOp>(loc, lower_bound,
                                                    upper_bound, step);

            for (Operation& nested : *loop.getBody()) {
                rewriter.eraseOp(&nested);
            }

            // 64 bits
            auto arith_index_i64 = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getI64Type(), loop.getInductionVar());

            loopIvs.push_back(arith_index_i64);
            rewriter.setInsertionPointToEnd(loop.getBody());

            if (i != e - 1) {
                rewriter.create<LLVM::CallOp>(loc, getScanfType(context),
                                              scanf_ref, newline_cast);
            }

            rewriter.create<scf::YieldOp>(loc);
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        auto lowered_memref = operands[0];
        mlir::MemRefDescriptor desc(lowered_memref);

        auto raw_ptr = desc.alignedPtr(rewriter, loc);
        auto elem_type = desc.getElementPtrType();
        auto offset = desc.offset(rewriter, loc);

        Value arith_linear_idx = std::move(offset);
        for (auto [pos, arith_index_i64] : llvm::enumerate(loopIvs)) {
            auto stride = desc.stride(rewriter, loc, pos);

            auto arith_muli_idx = rewriter.create<arith::MulIOp>(
                loc, rewriter.getI64Type(), arith_index_i64, stride);

            arith_linear_idx = rewriter.create<arith::AddIOp>(
                loc, rewriter.getI64Type(), arith_linear_idx, arith_muli_idx);
        }

        auto llvm_gep = rewriter.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(context), elem_type, raw_ptr,
            ArrayRef<Value>{arith_linear_idx});

        // llvm_gep->dump();

        rewriter.create<LLVM::CallOp>(
            loc, getScanfType(context), scanf_ref,
            ArrayRef<Value>({format_specifier_cast, llvm_gep}));

        rewriter.eraseOp(op);
        return llvm::success();
    }

  private:
    static LLVM::LLVMFunctionType getScanfType(MLIRContext* context) {
        auto llvm_i32Ty = IntegerType::get(context, 32);
        auto llvm_ptrTy = LLVM::LLVMPointerType::get(context);
        auto llvm_fnType =
            LLVM::LLVMFunctionType::get(llvm_i32Ty, llvm_ptrTy, true);

        return llvm_fnType;
    }

    static FlatSymbolRefAttr getOrInsertScanf(PatternRewriter& rewriter,
                                              ModuleOp module) {
        auto* context = module->getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>("scanf")) {
            return SymbolRefAttr::get(context, "scanf");
        }

        // insert the scanf function decl
        PatternRewriter::InsertionGuard insert_guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module->getLoc(), "scanf",
                                          getScanfType(context));

        return SymbolRefAttr::get(context, "scanf");
    }

    static Value getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                         StringRef name, StringRef value,
                                         ModuleOp module) {
        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
            OpBuilder::InsertionGuard insert_guard(builder);

            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(
                loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
                builder.getStringAttr(value),
                /*alignment=*/0);
        }

        // gep
        Value global_ptr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                      builder.getIndexAttr(0));

        return builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(builder.getContext()),
            global.getType(), global_ptr, ArrayRef<Value>({cst0, cst0}));
    }
};

} // namespace

// TToyLLVMLoweringPass
namespace {
struct TToyToLLVMLoweringPass
    : public PassWrapper<TToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TToyToLLVMLoweringPass)

    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }
    void runOnOperation() final;
};
} // namespace

void TToyToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    LLVMTypeConverter type_converter(&getContext());
    RewritePatternSet patterns(&getContext());

    // affine
    populateAffineToStdConversionPatterns(patterns);
    // scf
    populateSCFToControlFlowConversionPatterns(patterns);
    // arith type convert
    mlir::arith::populateArithToLLVMConversionPatterns(type_converter,
                                                       patterns);
    // memref type convert
    populateFinalizeMemRefToLLVMConversionPatterns(type_converter, patterns);
    // cf type convert
    mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                          patterns);
    populateFuncToLLVMConversionPatterns(type_converter, patterns);

    // scf style printop & scanop
    patterns.add<PrintOpLowering>(&getContext());
    patterns.add<ScanOpLowering>(&getContext());

    auto module = getOperation();
    if (llvm::failed(
            applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::ttoy::createLowerToLLVMPass() {
    return std::make_unique<TToyToLLVMLoweringPass>();
}