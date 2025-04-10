include "mlir/IR/OpBase.td"

def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
  let summary = "Toy Dialect";
}
