#!/usr/bin/env bash
set -euo pipefail
sdk=$(realpath "$1")
out=$(realpath -m "$2")
cd "$(dirname "$0")/.."
gen=$(mktemp -d)
trap 'rm -rf "$gen"' EXIT
ir=ttmlir/Dialect/EmitPy/IR
iface=ttmlir/Dialect/EmitPy/Interfaces
mkdir -p "$gen/$ir" "$gen/$iface"
table() {
    "$sdk/bin/mlir-tblgen" -I include -I "include/$ir" -I "include/$iface" -I "$sdk/include" "$@"
}
for mode in decls defs; do
    suffix=h
    if [[ $mode == defs ]]; then suffix=cpp; fi
    table "-gen-op-$mode" "include/$ir/EmitPyOps.td" -o "$gen/$ir/EmitPyOps.$suffix.inc"
    table "-gen-dialect-$mode" -dialect=emitpy "include/$ir/EmitPyOps.td" -o "$gen/$ir/EmitPyOpsDialect.$suffix.inc"
    table "-gen-typedef-$mode" -typedefs-dialect=emitpy "include/$ir/EmitPyOps.td" -o "$gen/$ir/EmitPyOpsTypes.$suffix.inc"
    table "-gen-attrdef-$mode" "include/$ir/EmitPyAttrs.td" -o "$gen/$ir/EmitPyAttrs.$suffix.inc"
    table "-gen-op-interface-$mode" "include/$iface/EmitPyInterfaces.td" -o "$gen/$iface/EmitPyInterfaces.$suffix.inc"
done
"${CXX:-clang++}" -std=c++17 -stdlib=libc++ -nostdinc++ -isystem "$sdk/include/aarch64-unknown-linux-gnu/c++/v1" -isystem "$sdk/include/c++/v1" -include cstddef -fno-rtti -O1 -B"${RIDGER_LLD_BIN:?set RIDGER_LLD_BIN}" -fuse-ld=lld -Wl,--threads=2 -Wl,--thinlto-jobs=2 -I include -I "$gen" -isystem "$sdk/include" \
    lib/Dialect/EmitPy/IR/*.cpp lib/Dialect/EmitPy/Interfaces/*.cpp \
    lib/Target/Python/TranslateToPython.cpp ridger/translate.cpp \
    -L "$sdk/lib" -L "$sdk/lib/aarch64-unknown-linux-gnu" -Wl,--start-group \
    -lMLIRIRDL -lMLIRFuncDialect -lMLIRParser -lMLIRAsmParser -lMLIRBytecodeReader \
    -lMLIRBytecodeWriter -lMLIRIR -lMLIRSupport -lMLIRSideEffectInterfaces \
    -lMLIRControlFlowInterfaces -lMLIRCallInterfaces -lMLIRCastInterfaces \
    -lMLIRFunctionInterfaces -lMLIRInferTypeOpInterface \
    -lLLVMSupport -lLLVMDemangle -Wl,--end-group -lc++abi -lunwind -lz -lzstd -ldl -pthread \
    -Wl,-rpath,"${RIDGER_EMITPY_LIBS:?set RIDGER_EMITPY_LIBS}" -o "$out"
