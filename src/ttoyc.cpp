//===- ttoyc.cpp - The ttoy Compiler
//----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the ttoy compiler.
//
//===----------------------------------------------------------------------===//

#include "parser/AST.h"
#include "parser/Lexer.h"
#include "parser/Parser.h"
#include "pass/Passes.hpp"
#include "ttoy/Dialect.hpp"
#include "ttoy/IRGen.hpp"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FileSystem.h"

#include <cstdlib>
#include <format>
#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>

#include <mlir/Transforms/Passes.h>

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace ttoy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ttoy file>"),
                                          cl::init("-"),
                                          cl::value_desc("input filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("<output ttoy file>"),
                                           cl::init("-"),
                                           cl::value_desc("output filename"));

namespace {
enum InputType { TToy, MLIR };
} // namespace
static cl::opt<enum InputType>
    inputType("x", cl::init(TToy),
              cl::desc("Decided the kind of output desired"),
              cl::values(clEnumValN(TToy, "ttoy",
                                    "load the input file as a TToy source.")),
              cl::values(clEnumValN(MLIR, "mlir",
                                    "load the input file as an MLIR file")));

namespace {
enum Action {
    None,
    DumpAST,
    DumpMLIR,
    DumpMLIRAffine,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT,
    BuildExe,
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")),
    cl::values(clEnumValN(BuildExe, "exe",
                          "Build objfile and link as an exe")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a TToy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<ttoy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext& context,
             mlir::OwningOpRef<mlir::ModuleOp>& module) {
    // Handle '.toy' input to the compiler.
    if (inputType != InputType::MLIR &&
        !llvm::StringRef(inputFilename).ends_with(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        module = mlirGen(context, *moduleAST);
        return !module ? 1 : 0;
    }

    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext& context,
                       mlir::OwningOpRef<mlir::ModuleOp>& module) {
    if (int error = loadMLIR(context, module))
        return error;

    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
        return 4;

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (enableOpt || isLoweringToAffine) {
        // Inline all functions into main and then delete them.
        pm.addPass(mlir::createInlinerPass());

        // Now that there is only one function, we can infer the shapes of each
        // of the operations.
        mlir::OpPassManager& optPM = pm.nest<mlir::ttoy::FuncOp>();
        optPM.addPass(mlir::ttoy::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }

    if (isLoweringToAffine) {
        // Partially lower the toy dialect.
        pm.addPass(mlir::ttoy::createLowerToAffinePass());

        // Add a few cleanups post lowering.
        mlir::OpPassManager& optPM = pm.nest<mlir::func::FuncOp>();
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());

        // Add optimizations if enabled.
        if (enableOpt) {
            optPM.addPass(mlir::affine::createLoopFusionPass());
            optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
        }
    }

    if (isLoweringToLLVM) {
        // Finish lowering the toy IR to the LLVM dialect.
        pm.addPass(mlir::ttoy::createLowerToLLVMPass());
        // This is necessary to have line tables emitted and basic
        // debugger working. In the future we will add proper debug information
        // emission directly from our frontend.
        pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
    }

    if (mlir::failed(pm.run(*module)))
        return 4;
    return 0;
}

int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;

    dump(*moduleAST);
    return 0;
}

int dumpMLIR(mlir::ModuleOp module) {
    // Action::DumpMLIR DumpMLIRAffine DumpMLIRLLVM

    std::error_code ec;
    llvm::raw_fd_ostream dst(outputFilename, ec, llvm::sys::fs::OF_None);

    if (ec) {
        llvm::errs() << "Failed to open an output file\n";
        return -1;
    }

    module->print(dst);
    dst.flush();

    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Configure the LLVM Module
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        llvm::errs() << "Could not create JITTargetMachineBuilder\n";
        return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        llvm::errs() << "Could not create TargetMachine\n";
        return -1;
    }
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
        llvmModule.get(), tmOrError.get().get());

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    std::error_code ec;
    llvm::raw_fd_ostream dst(outputFilename, ec, llvm::sys::fs::OF_None);

    if (ec) {
        llvm::errs() << "Failed to open output file\n";
        return -1;
    }

    llvmModule->print(dst, nullptr);
    dst.flush();

    return 0;
}

int runJit(mlir::ModuleOp module) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before
    // we can JIT-compile.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // An optimization pipeline to use within the execution engine.
    auto opt_pipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly
    // JIT-compiles the module.
    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = opt_pipeline;
    auto maybe_engine = mlir::ExecutionEngine::create(module, engine_options);
    assert(maybe_engine && "failed to construct an execution engine");
    auto& engine = maybe_engine.get();

    // Invoke the JIT-compiled function.
    auto invocation_result = engine->invokePacked("main");
    if (invocation_result) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int BuildExecutable(mlir::ModuleOp module) {
    mlir::registerBuiltinDialectTranslation(*module.getContext());
    mlir::registerLLVMDialectTranslation(*module.getContext());

    llvm::LLVMContext llvm_context;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    if (!llvm_module) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    /// TODO:

    llvm::StringRef triple_str = LLVMGetDefaultTargetTriple();
    llvm_module->setTargetTriple(triple_str);

    std::string err;
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple_str, err);

    llvm::TargetOptions opts; // empty
    std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
        triple_str, "generic", "", opts, std::nullopt));

    llvm_module->setDataLayout(tm->createDataLayout());

    std::error_code ec;

    // an obj file
    llvm::raw_fd_ostream dst("out.o", ec, llvm::sys::fs::OF_None);

    if (ec) {
        llvm::errs() << "Failed to open file for the obj: " + ec.message() +
                            '\n';
        return -2;
    }

    llvm::legacy::PassManager pass;
    if (tm->addPassesToEmitFile(pass, dst, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
        llvm::errs() << "Failed to generate obj file\n";
        return -3;
    }

    pass.run(*llvm_module);

    dst.flush();

    /// TODO: replace clang++ with native linkers
    // link
    auto ret0 = std::system(
        std::format("clang++ out.o -o {}",
                    outputFilename == "-" ? "a.out" : outputFilename.c_str())
            .c_str());

    auto ret1 = std::system("rm out.o");

    if (ret0) {
        llvm::errs() << "Failed to link object file \n";
        return -4;
    }

    if (ret1) {
        llvm::errs() << "Failed to rm object file\n";
        return -5;
    }

    return 0;
}

int main(int argc, char** argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "ttoy compiler\n");

    if (emitAction == Action::DumpAST)
        return dumpAST();

    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    mlir::LLVM::registerInlinerInterface(registry);

    mlir::MLIRContext context(registry);
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::ttoy::TToyDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;

    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR) {
        return dumpMLIR(*module);
    }

    if (emitAction == Action::DumpLLVMIR) {
        return dumpLLVMIR(*module);
    }

    if (emitAction == Action::RunJIT) {
        return runJit(*module);
    }

    if (emitAction == Action::BuildExe) {
        return BuildExecutable(*module);
    }

    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
