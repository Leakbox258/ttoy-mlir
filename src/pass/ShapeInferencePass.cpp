//===- ShapeInferenceOpInterface.cpp - Shape Inference
//---------------------------===//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//
#include "pass/Passes.hpp"
#include "pass/ShapeInferenceOpInterface.hpp"
#include "ttoy/Dialect.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

#include <memory>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace ttoy;

#include "generated/ShapeInferenceOpInterface.cpp.inc"

namespace {

// pass run on function
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass,
                               OperationPass<ttoy::FuncOp>> {
    // explict pass id
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

    // pass impl
    void runOnOperation() override {
        auto f = getOperation();

        llvm::SmallPtrSet<mlir::Operation*, 16> op_work_list;

        // build worklist
        f->walk([&](mlir::Operation* op) {
            if (returnsDynamicShape(op)) {
                op_work_list.insert(op);
            }
        });

        while (!op_work_list.empty()) {
            // find an operation which inputs are inferred
            auto next_op = llvm::find_if(op_work_list, allOperansInferred);

            if (next_op == op_work_list.end()) {
                break;
            }

            Operation* op = *next_op;
            op_work_list.erase(op);

            LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
            if (auto shape_op = llvm::dyn_cast<ShapeInference>(op)) {
                shape_op.inferShapes();
            } else {
                op->emitError(
                    "unable to infer shape of operation without shape "
                    "inference interface");
                return signalPassFailure();
            }
        }

        if (!op_work_list.empty()) {
            f.emitError("Shape inference failed, ")
                << op_work_list.size() << " operations couldn't be inferred\n";
            signalPassFailure();
        }
    }

    // op->getResultTypes() all unranked
    static bool returnsDynamicShape(Operation* op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            return !llvm::isa<RankedTensorType>(resultType);
        });
    }

    static bool allOperansInferred(Operation* op) {
        return llvm::any_of(op->getOperandTypes(), [](Type operandType) {
            return !llvm::isa<RankedTensorType>(operandType);
        });
    }
};

} // namespace

// pass create inferface
std::unique_ptr<mlir::Pass> mlir::ttoy::createShapeInferencePass() {
    return std::make_unique<ShapeInferencePass>();
}