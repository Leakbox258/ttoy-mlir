//====- LowerToAffine.cpp - Partial lowering from TToy to Affine+Std --===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of TToy operations to a combination
// of affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <numeric>
#include <pass/Passes.hpp>
#include <ttoy/Dialect.hpp>

/// Generics
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

/// Builtin dialect
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

/// MLIR utils
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

/// LLVM utils
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/Support/Casting.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

static MemRefType convertTensorToMemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

static memref::AllocOp insertAllocAndDealloc(MemRefType type, Location loc,
                                             PatternRewriter& rewriter) {
    // 1
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);

    auto* parent_block = alloc->getBlock();

    alloc->moveBefore(&parent_block->front());

    // 2
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parent_block->back());

    return alloc;
}

using LoopIterFn = function_ref<Value(
    OpBuilder& rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation* op, ValueRange operands,
                           PatternRewriter& rewriter, LoopIterFn process_iter) {
    auto tensor_type = llvm::cast<RankedTensorType>(*op->result_type_begin());

    auto loc = op->getLoc();

    // alloc: get a memref & auto dealloc
    auto memref_type = convertTensorToMemRef(tensor_type);
    auto alloc = insertAllocAndDealloc(memref_type, loc, rewriter);

    llvm::SmallVector<int64_t, 4> lower_bounds(tensor_type.getRank());
    llvm::SmallVector<int64_t, 4> steps(tensor_type.getRank(), 1);

    affine::buildAffineLoopNest(
        rewriter, loc, lower_bounds, tensor_type.getShape(), steps,
        [&](OpBuilder& nested_builder, Location loc, ValueRange ivs) {
            Value value_to_store = process_iter(nested_builder, operands, ivs);

            nested_builder.create<affine::AffineStoreOp>(loc, value_to_store,
                                                         alloc, ivs);
        });

    rewriter.replaceOp(op, alloc);
}

static void clearAllocOp(memref::AllocOp op, OpBuilder& builder, Location loc) {
    auto shape = op.getType().getShape();
    auto size = std::accumulate(
        shape.begin(), shape.end(), 8ll /* sizeof f64 */,
        [&](const auto& acc, const auto& dim) { return acc * dim; });

    auto rawPtrIndex =
        builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, op);
    auto castI64 = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                      rawPtrIndex);

    auto llvm_PtrTy = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    auto llvmInt2Ptr =
        builder.create<LLVM::IntToPtrOp>(loc, llvm_PtrTy, castI64);

    // memset intrinsic: void llvm.memset.p0.i64(i8* ptr, i8 value, i64 len, i1
    // isVolatile)
    auto cst0_i8 =
        builder.create<arith::ConstantIntOp>(loc, 0, builder.getI8Type());
    auto cst_size =
        builder.create<arith::ConstantIntOp>(loc, size, builder.getI32Type());
    auto cstFalse =
        builder.create<arith::ConstantIntOp>(loc, 0, builder.getI1Type());

    builder.create<LLVM::MemsetOp>(loc, llvmInt2Ptr, cst0_i8, cst_size,
                                   cstFalse);
}

namespace {

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext* ctx)
        : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(
            op, operands, rewriter,
            [loc](OpBuilder& builder, ValueRange memRefOperands,
                  ValueRange loopIvs) -> Value {
                // Generate an adaptor for the remapped operands of the
                // BinaryOp. This allows for using the nice named accessors
                // that are generated by the ODS.
                typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                // Generate loads for the element of 'lhs' and 'rhs' at the
                // inner loop.
                auto loadedLhs = builder.create<affine::AffineLoadOp>(
                    loc, binaryAdaptor.getLhs(), loopIvs);
                auto loadedRhs = builder.create<affine::AffineLoadOp>(
                    loc, binaryAdaptor.getRhs(), loopIvs);

                // Create the binary operation performed on the loaded
                // values.
                return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                       loadedRhs);
            });
        return success();
    }
};
using AddOpLowering = BinaryOpLowering<ttoy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<ttoy::MulOp, arith::MulFOp>;
using SubOpLowering = BinaryOpLowering<ttoy::SubOp, arith::SubFOp>;
using DivOpLowering = BinaryOpLowering<ttoy::DivOp, arith::DivFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<ttoy::ConstantOp> {
    using OpRewritePattern<ttoy::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ttoy::ConstantOp op,
                                  PatternRewriter& rewriter) const final {
        DenseElementsAttr constantValue = op.getValue();
        Location loc = op.getLoc();

        // When lowering the constant operation, we allocate and assign the
        // constant values to a corresponding memref allocation.
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // We will be generating constant indices up-to the largest dimension.
        // Create these constants up-front to avoid large amounts of redundant
        // operations.
        auto valueShape = memRefType.getShape();
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
            for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
                constantIndices.push_back(
                    rewriter.create<arith::ConstantIndexOp>(loc, i));
        } else {
            // This is the case of a tensor of rank 0.
            constantIndices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, 0));
        }

        // The constant operation represents a multi-dimensional constant, so we
        // will need to generate a store for each of the elements. The following
        // functor recursively walks the dimensions of the constant shape,
        // generating a store when the recursion hits the base case.
        SmallVector<Value, 2> indices;
        auto valueIt = constantValue.value_begin<FloatAttr>();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            // The last dimension is the base case of the recursion, at this
            // point we store the element at the given index.
            if (dimension == valueShape.size()) {
                rewriter.create<affine::AffineStoreOp>(
                    loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++),
                    alloc, llvm::ArrayRef(indices));
                return;
            }

            // Otherwise, iterate over the current dimension and add the indices
            // to the list.
            for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        // Start the element storing recursion from the first dimension.
        storeElements(/*dimension=*/0);

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<ttoy::FuncOp> {
    using OpConversionPattern<ttoy::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ttoy::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter& rewriter) const final {
        // We only lower the main function as we expect that all other functions
        // have been inlined.
        if (op.getName() != "main")
            return failure();

        // Verify that the given main has no inputs and results.
        if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
                diag << "expected 'main' to have 0 inputs and 0 results";
            });
        }

        // Create a new non-toy function, with the same region.
        auto func = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), op.getName(), op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<ttoy::PrintOp> {
    using OpConversionPattern<ttoy::PrintOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ttoy::PrintOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter& rewriter) const final {
        // We don't lower "toy.print" in this pass, but we need to update its
        // operands.
        rewriter.modifyOpInPlace(
            op, [&] { op->setOperands(adaptor.getOperands()); });
        return llvm::success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Scan operations
//===----------------------------------------------------------------------===//

struct ScanOpLowering : public OpConversionPattern<ttoy::ScanOp> {
    using OpConversionPattern<ttoy::ScanOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ttoy::ScanOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter& rewriter) const final {
        rewriter.modifyOpInPlace(
            op, [&] { op->setOperands(adaptor.getOperands()); });
        return llvm::success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<ttoy::ReturnOp> {
    using OpRewritePattern<ttoy::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ttoy::ReturnOp op,
                                  PatternRewriter& rewriter) const final {
        // During this lowering, we expect that all function calls have been
        // inlined.
        if (op.hasOperand())
            return failure();

        // We lower "toy.return" directly to "func.return".
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
    TransposeOpLowering(MLIRContext* ctx)
        : ConversionPattern(ttoy::TransposeOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(
            op, operands, rewriter,
            [loc](OpBuilder& builder, ValueRange memRefOperands,
                  ValueRange loopIvs) {
                // Generate an adaptor for the remapped operands of the
                // TransposeOp. This allows for using the nice named
                // accessors that are generated by the ODS.
                ttoy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                Value input = transposeAdaptor.getInput();

                // Transpose the elements by generating a load from the
                // reverse indices.
                SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                return builder.create<affine::AffineLoadOp>(loc, input,
                                                            reverseIvs);
            });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Dot operations
//===----------------------------------------------------------------------===//
struct DotOpLowering : public ConversionPattern {
    DotOpLowering(MLIRContext* ctx)
        : ConversionPattern(ttoy::DotOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();

        auto tensor_type =
            llvm::cast<RankedTensorType>(*op->result_type_begin());
        auto memref_type = convertTensorToMemRef(tensor_type);
        auto alloc = insertAllocAndDealloc(memref_type, loc, rewriter);

        clearAllocOp(alloc, rewriter, loc);

        llvm::SmallVector<int64_t, 2> lower_bounds(tensor_type.getRank()); // 1
        llvm::SmallVector<int64_t, 2> steps(tensor_type.getRank(), 1);

        ttoy::MMOpAdaptor adaptor(operands);

        /// lhs will lowering before dot, so here is MemRefType, I guess...
        const auto lhs_shape =
            llvm::dyn_cast<MemRefType>(adaptor.getLhs().getType()).getShape();

        llvm::SmallVector<int64_t, 2> upper_bounds{lhs_shape[0]};

        auto cst0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        affine::buildAffineLoopNest(
            rewriter, loc, lower_bounds, upper_bounds, steps,
            [&](OpBuilder& nested_builder, Location loc, ValueRange ivs) {
                ttoy::DotOp::Adaptor adaptor(operands);

                // lhs through idx
                auto load_lhs = nested_builder.create<affine::AffineLoadOp>(
                    loc, adaptor.getLhs(), ivs);

                // rhs through idx
                auto load_rhs = nested_builder.create<affine::AffineLoadOp>(
                    loc, adaptor.getRhs(), ivs);

                auto mul = nested_builder.create<arith::MulFOp>(loc, load_lhs,
                                                                load_rhs);

                auto load_res = nested_builder.create<affine::AffineLoadOp>(
                    loc, alloc, ValueRange{cst0});

                auto add_res =
                    nested_builder.create<arith::AddFOp>(loc, mul, load_res);

                nested_builder.create<affine::AffineStoreOp>(
                    loc, add_res, alloc, ValueRange{cst0});
            });

        rewriter.replaceOp(op, alloc);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: MM operations
//===----------------------------------------------------------------------===//
struct MMOpLowering : public ConversionPattern {
    MMOpLowering(MLIRContext* ctx)
        : ConversionPattern(ttoy::MMOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        auto tensor_type =
            llvm::dyn_cast<RankedTensorType>(*op->result_type_begin());
        auto memref_type = convertTensorToMemRef(tensor_type);

        auto alloc = insertAllocAndDealloc(memref_type, loc, rewriter);

        clearAllocOp(alloc, rewriter, loc);

        llvm::SmallVector<int64_t, 4> lower_bounds(tensor_type.getRank() + 1);
        llvm::SmallVector<int64_t, 4> steps(tensor_type.getRank() + 1, 1);

        ttoy::MMOpAdaptor adaptor(operands);

        const auto lhs_shape =
            llvm::dyn_cast<MemRefType>(adaptor.getLhs().getType()).getShape();
        const auto rhs_shape =
            llvm::dyn_cast<MemRefType>(adaptor.getRhs().getType()).getShape();

        llvm::SmallVector<int64_t, 4> upper_bounds{lhs_shape[0], rhs_shape[1],
                                                   lhs_shape[1]};

        affine::buildAffineLoopNest(
            rewriter, loc, lower_bounds, upper_bounds, steps,
            [&](OpBuilder& nested_builder, Location loc, ValueRange ivs_i_j_k) {
                llvm::SmallVector<Value, 2> ivs_i_k{ivs_i_j_k[0], ivs_i_j_k[2]};
                llvm::SmallVector<Value, 2> ivs_k_j{ivs_i_j_k[2], ivs_i_j_k[1]};
                llvm::SmallVector<Value, 2> ivs_i_j{ivs_i_j_k[0], ivs_i_j_k[1]};

                auto load_lhs = nested_builder.create<affine::AffineLoadOp>(
                    loc, adaptor.getLhs(), ivs_i_k);
                auto load_rhs = nested_builder.create<affine::AffineLoadOp>(
                    loc, adaptor.getRhs(), ivs_k_j);

                auto mul = nested_builder.create<arith::MulFOp>(loc, load_lhs,
                                                                load_rhs);

                auto load_res = nested_builder.create<affine::AffineLoadOp>(
                    loc, alloc, ivs_i_j);

                auto add_res =
                    nested_builder.create<arith::AddFOp>(loc, mul, load_res);

                nested_builder.create<affine::AffineStoreOp>(loc, add_res,
                                                             alloc, ivs_i_j);
            });

        rewriter.replaceOp(op, alloc);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: BMM operations
//===----------------------------------------------------------------------===//

struct BMMOpLowering : public ConversionPattern {
    BMMOpLowering(MLIRContext* ctx)
        : ConversionPattern(ttoy::BMMOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                    ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        auto tensor_type =
            llvm::dyn_cast<RankedTensorType>(*op->result_type_begin());
        auto memref_type = convertTensorToMemRef(tensor_type);

        auto alloc = rewriter.create<memref::AllocOp>(loc, memref_type);

        clearAllocOp(alloc, rewriter, loc);

        llvm::SmallVector<int64_t, 4> lower_bounds(tensor_type.getRank() + 1);
        llvm::SmallVector<int64_t, 4> steps(tensor_type.getRank() + 1, 1);

        ttoy::BMMOpAdaptor adaptor(operands);

        const auto lhs_shape =
            llvm::dyn_cast<MemRefType>(adaptor.getLhs().getType()).getShape();
        const auto rhs_shape =
            llvm::dyn_cast<MemRefType>(adaptor.getRhs().getType()).getShape();
        llvm::SmallVector<int64_t, 4> upper_bounds{tensor_type.getShape()[0],
                                                   lhs_shape[0], rhs_shape[1],
                                                   lhs_shape[1]};

        affine::buildAffineLoopNest(
            rewriter, loc, lower_bounds, upper_bounds, steps,
            [&](OpBuilder& nested_builder, Location loc,
                ValueRange ivs_b_i_j_k) {
                llvm::SmallVector<Value, 4> ivs_b_i_k{
                    ivs_b_i_j_k[0], ivs_b_i_j_k[1], ivs_b_i_j_k[3]};
                llvm::SmallVector<Value, 4> ivs_b_k_j{
                    ivs_b_i_j_k[0], ivs_b_i_j_k[3], ivs_b_i_j_k[2]};
                llvm::SmallVector<Value, 4> ivs_b_i_j{
                    ivs_b_i_j_k[0], ivs_b_i_j_k[1], ivs_b_i_j_k[2]};

                auto load_lhs = nested_builder.create<affine::AffineLoadOp>(
                    loc, adaptor.getLhs(), ivs_b_i_k);
                auto load_rhs = nested_builder.create<affine::AffineLoadOp>(
                    loc, adaptor.getRhs(), ivs_b_k_j);

                auto mul = nested_builder.create<arith::MulFOp>(loc, load_lhs,
                                                                load_rhs);

                auto load_res = nested_builder.create<affine::AffineLoadOp>(
                    loc, alloc, ivs_b_i_j);

                auto add_res =
                    nested_builder.create<arith::AddFOp>(loc, mul, load_res);

                nested_builder.create<affine::AffineStoreOp>(loc, add_res,
                                                             alloc, ivs_b_i_j);
            });

        rewriter.replaceOp(op, alloc);

        return success();
    }
};

} // namespace

// lowering pass
namespace {
struct TToyToAffineLoweringPass
    : public PassWrapper<TToyToAffineLoweringPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TToyToAffineLoweringPass)

    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<affine::AffineDialect, func::FuncDialect,
                        LLVM::LLVMDialect, memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
} // namespace

void TToyToAffineLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                           arith::ArithDialect, func::FuncDialect,
                           LLVM::LLVMDialect, memref::MemRefDialect>();
    target.addIllegalDialect<ttoy::TToyDialect>();

    // specifically handling PrintOp & ScanOp
    target.addDynamicallyLegalOp<ttoy::PrintOp>([](ttoy::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(), [](Type type) {
            return llvm::isa<TensorType>(type);
        });
    });
    target.addDynamicallyLegalOp<ttoy::ScanOp>([](ttoy::ScanOp op) {
        return llvm::none_of(op->getOperandTypes(), [](Type type) {
            return llvm::isa<TensorType>(type);
        });
    });

    // the set of patterns that will lower the TToy operations
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, SubOpLowering, ConstantOpLowering,
                 FuncOpLowering, MulOpLowering, DivOpLowering, DotOpLowering,
                 MMOpLowering, BMMOpLowering, PrintOpLowering, ScanOpLowering,
                 ReturnOpLowering, TransposeOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
        signalPassFailure();
    }
}

// interface
std::unique_ptr<Pass> mlir::ttoy::createLowerToAffinePass() {
    return std::make_unique<TToyToAffineLoweringPass>();
}