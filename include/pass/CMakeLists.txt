set(LLVM_TARGET_DEFINITIONS 
    PatternMatch.td    
)

mlir_tablegen(PatternMatch.inc -gen-rewriters)

add_public_tablegen_target(TToy_PatternMatch)

add_custom_command(
  TARGET TToy_PatternMatch POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_BINARY_DIR}/*.inc
          ${TABLEGEN_OUTPUT_DIR}/.
)


message(STATUS "MLIR pattern matches TB files configured.")