//===- Ops.td - TToy dialect operations definitions ----------*- tablegen -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the TToy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TTOY_OPERATIONS
#define TTOY_OPERATIONS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

include "pass/ShapeInferenceOpInterface.td"

// dialect definitions
def TToy_Dialect : Dialect {
    let name = "ttoy";
    let cppNamespace = "::mlir::ttoy";
    let summary = "a tensor based ttoy dialect";
    let description = [{
        mlir extension.
    }];
    // TODO
}

class TToy_Op<string mnemonic, list<Trait> traits = []> : Op<TToy_Dialect, mnemonic, traits>;

// ConstantOp
// this how ttoy represent a constant tensor
// input a descriptive attribute and output a tensor
// the attr and value type is provided by mlir infra
def ConstantOp : TToy_Op<"constant", [Pure]> {
    // will print as 'ttoy.constant'
    let summary = "constant";
    let arguments = (ins 
        F64ElementsAttr : $value
    );
    let results = (outs 
        F64Tensor
    );
    
    // to custom parse & print method 
    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "DenseElementsAttr" : $value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,

        OpBuilder<(ins "double": $value)>
    ];

    let hasVerifier = 1;
}

// AddOp
def AddOp : TToy_Op<"add", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise add";
    let arguments = (ins 
        F64Tensor : $lhs, 
        F64Tensor : $rhs
    );
    let results = (outs 
        F64Tensor
    );

    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>
    ];
}

// SubOp
def SubOp : TToy_Op<"sub", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise sub";
    let arguments = (ins
        F64Tensor : $lhs,
        F64Tensor : $rhs
    );
    let results = (outs
        F64Tensor
    );

    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>
    ];
}

// MulOp
def MulOp : TToy_Op<"mul", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise mul";
    let arguments = (ins
        F64Tensor : $lhs,
        F64Tensor : $rhs
    );
    let results = (outs
        F64Tensor
    );
    
    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins
            "Value" : $lhs,
            "Value" : $rhs  
        )>
    ];
}

// DivOp
def DivOp : TToy_Op<"div", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise Div";
    let arguments = (ins
        F64Tensor : $lhs,
        F64Tensor : $rhs
    );
    let results = (outs
        F64Tensor
    );
    
    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins
            "Value" : $lhs,
            "Value" : $rhs  
        )>
    ];
}

// FuncOp
def FuncOp : TToy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef" : $name,
    "FunctionType" : $type,
    CArg<"ArrayRef<NamedAttribute>", "{}"> : $attrs)
  >];

  let extraClassDeclaration = [{

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

// CallOp
def CallOp : TToy_Op<"call",
[DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "call operation";
    let arguments = (ins
        FlatSymbolRefAttr : $callee,
        Variadic<F64Tensor> : $inputs  
    );
    let results = (outs
        F64Tensor
    );
    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];
    let builders = [
        OpBuilder<(ins
            "StringRef" : $callee,
            "ArrayRef<Value>" : $arguments
        )>
    ];
}

// ReturnOp
def ReturnOp : TToy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
    let summary = "return operation";
    let arguments = (ins
        Variadic<F64Tensor> : $input
    );

    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, std::nullopt);
        }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    // Invoke a static verify method to verify this return operation.
    let hasVerifier = 1;
}

///  built-in functions

def PrintOp : TToy_Op<"print"> {
    let summary = "print builtin operation";
    let arguments = (ins
        AnyTypeOf<[F64Tensor, F64MemRef]> : $input
    );
    
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp : TToy_Op<"reshape"> {
    let summary = "tensor-reshape builtin operation";
    let arguments = (ins 
        F64Tensor : $input
    );

    let results = (outs
        StaticShapeTensorOf<[F64]>
    );
    
    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)        
    }];

    let hasCanonicalizer = 1;
}

def TransposeOp : TToy_Op<"transpose", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "tensor-transpose builtin operation";
    
    let arguments = (ins
        F64Tensor : $input
    );
    let results = (outs
        F64Tensor
    );

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];
    let builders = [
        OpBuilder<(ins
            "Value" : $input
        )>];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def CastOp : TToy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    Pure,
    SameOperandsAndResultShape    
]> {
    let summary = "shape cast operation";
    
    let arguments = (ins F64Tensor : $input);
    let results = (outs F64Tensor : $output);

    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

#endif // 