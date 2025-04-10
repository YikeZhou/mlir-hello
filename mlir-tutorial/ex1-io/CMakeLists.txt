add_executable(ex1-io main.cpp)

target_link_libraries(
  ex1-io
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)