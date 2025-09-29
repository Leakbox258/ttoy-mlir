set(LLVM_TARGET_DEFINITIONS 
    Dialect.td
)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_public_tablegen_target(Etoy_Dialect_Operations)

add_custom_command(
  TARGET Etoy_Dialect_Operations POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_BINARY_DIR}/*.inc
          ${TABLEGEN_OUTPUT_DIR}/.
)


message(STATUS "MLIR dialect & operations configured.")