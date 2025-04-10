add_executable(ex2-build main.cpp)

target_link_libraries(
  ex2-build
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)