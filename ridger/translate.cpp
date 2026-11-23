// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Target/Python/PythonEmitter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace mlir;
namespace py = mlir::tt::emitpy;

namespace {
struct Lowering {
  OpBuilder &builder;
  Location loc;
  Type object;

  Value call(StringRef name, ValueRange operands, ArrayRef<Attribute> args = {},
             ArrayRef<Attribute> keywords = {}) {
    auto arguments = args.empty() ? ArrayAttr() : builder.getArrayAttr(args);
    auto names = args.empty() ? ArrayAttr() : builder.getArrayAttr(keywords);
    return builder.create<py::CallOpaqueOp>(loc, TypeRange{object}, name,
                                            operands, arguments, names)
        .getResult(0);
  }

  Value path(ArrayRef<StringRef> segments) {
    SmallVector<Attribute> args, names;
    for (auto segment : segments) {
      args.push_back(builder.getStringAttr(segment));
      names.push_back(builder.getStringAttr(""));
    }
    return call("util_create_list", {}, args, names);
  }

  Value lookup(ValueRange args, bool strict) {
    auto none = builder.getStringAttr("");
    return call("_toolz.get_in", {args[1], args[0]},
                {builder.getIndexAttr(0), builder.getIndexAttr(1),
                 builder.getI64IntegerAttr(-17), builder.getBoolAttr(strict)},
                {none, none, builder.getStringAttr("default"),
                 builder.getStringAttr("no_default")});
  }

  Value increment(Value root) {
    auto blank = builder.getStringAttr("");
    auto operators = call("__import__", {}, {builder.getStringAttr("operator")}, {blank});
    auto add = call("getattr", {operators},
                    {builder.getIndexAttr(0), builder.getStringAttr("add")},
                    {blank, blank});
    auto plusOne = call("_functools.partial", {add},
                        {builder.getIndexAttr(0), builder.getI64IntegerAttr(1)},
                        {blank, blank});
    return call("_toolz.update_in", {root, path({"x", "y"}), plusOne},
                {builder.getIndexAttr(0), builder.getIndexAttr(1),
                 builder.getIndexAttr(2), builder.getI64IntegerAttr(0)},
                {blank, blank, blank, builder.getStringAttr("default")});
  }
};

bool hasTypes(TypeRange types, ArrayRef<StringRef> names) {
  if (types.size() != names.size())
    return false;
  for (auto [type, name] : llvm::zip(types, names)) {
    auto dynamic = dyn_cast<DynamicType>(type);
    if (!dynamic || dynamic.getTypeDef()->getName() != name ||
        dynamic.getDialect().getNamespace() != "d02")
      return false;
  }
  return true;
}

LogicalResult lower(ModuleOp input, ModuleOp output, unsigned index) {
  auto fn = input.lookupSymbol<func::FuncOp>("candidate");
  if (!fn || fn.isDeclaration() || !llvm::hasSingleElement(fn.getBody()) ||
      !hasTypes(fn.getArgumentTypes(), {"dict", "key", "value", "dict"}) ||
      !hasTypes(fn.getResultTypes(), {"value"}))
    return input.emitError("unsupported D02 candidate signature or control flow");
  for (Operation &top : *input.getBody())
    if (&top != fn.getOperation() &&
        (!isa<func::FuncOp>(top) || !cast<func::FuncOp>(top).isDeclaration()))
      return top.emitError("unsupported top-level operation");

  OpBuilder b(output.getBodyRegion());
  b.setInsertionPointToEnd(output.getBody());
  auto obj = py::OpaqueType::get(output.getContext(), "object");
  auto function = b.create<func::FuncOp>(
      fn.getLoc(), "route" + std::to_string(index),
      b.getFunctionType({obj, obj, obj, obj}, {obj, obj}));
  auto *block = function.addEntryBlock();
  b.setInsertionPointToStart(block);
  DenseMap<Value, Value> values;
  for (auto [source, target] : llvm::zip(fn.getArguments(), block->getArguments()))
    values[source] = target;
  Value pair;
  SmallVector<Value, 2> fields;
  for (Operation &op : fn.getBody().front()) {
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      if (!pair || ret.getNumOperands() != 1 || ret.getOperand(0) != pair)
        return op.emitError("expected return of the observed pair");
      b.create<func::ReturnOp>(op.getLoc(), fields);
      continue;
    }
    auto call = dyn_cast<func::CallOp>(op);
    if (!call || pair || call.getNumResults() != 1)
      return op.emitError("unsupported operation in D02 call sequence");
    auto declaration = input.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!declaration || !declaration.isDeclaration())
      return op.emitError("expected a declared D02 public-call adapter");
    SmallVector<Value> operands;
    for (auto value : call.getOperands()) {
      auto target = values.lookup(value);
      if (!target)
        return op.emitError("unsupported SSA operand");
      operands.push_back(target);
    }

    Lowering adapter{b, op.getLoc(), obj};
    auto types = call->getOperandTypes();
    auto results = call->getResultTypes();
    StringRef name = call.getCallee();
    Value result;
    if ((name == "get_default" || name == "get_strict") &&
        hasTypes(types, {"dict", "key"}) && hasTypes(results, {"value"})) {
      result = adapter.lookup(operands, name == "get_strict");
    } else if (name == "observe_pair" && hasTypes(types, {"value", "value"}) &&
               hasTypes(results, {"value"})) {
      pair = call.getResult(0);
      fields.assign(operands.begin(), operands.end());
      continue;
    } else if (name == "inc_y" && hasTypes(types, {"dict"}) &&
               hasTypes(results, {"dict"})) {
      result = adapter.increment(operands[0]);
    } else if (hasTypes(results, {"dict"}) &&
               (((name == "write_y" || name == "write_z" ||
                  name == "write_parent_scalar") && hasTypes(types, {"dict", "value"})) ||
                (name == "write_parent_map" && hasTypes(types, {"dict", "dict"})))) {
      Value path = name == "write_y" ? adapter.path({"x", "y"})
                   : name == "write_z" ? adapter.path({"x", "z"})
                                       : adapter.path({"x"});
      result = adapter.call("_toolz.assoc_in", {operands[0], path, operands[1]});
    } else {
      return op.emitError("unsupported D02 call or signature: ") << name;
    }
    values[call.getResult(0)] = result;
  }
  return success();
}
} // namespace

namespace {
int translate(MLIRContext &context, std::unique_ptr<llvm::MemoryBuffer> buffer,
              llvm::raw_ostream &destination) {
  using Clock = std::chrono::steady_clock;
  using Nanoseconds = std::chrono::nanoseconds;
  const auto started = Clock::now();
  OpBuilder b(&context);
  auto loc = b.getUnknownLoc();
  OwningOpRef<ModuleOp> output(ModuleOp::create(loc));
  b.setInsertionPointToStart(output->getBody());
  OperationState import(loc, py::ImportOp::getOperationName());
  import.addAttribute("module_name", b.getStringAttr("toolz.dicttoolz"));
  import.addAttribute("module_alias", b.getStringAttr("_toolz"));
  b.create(import);
  OperationState helperImport(loc, py::ImportOp::getOperationName());
  helperImport.addAttribute("module_name", b.getStringAttr("functools"));
  helperImport.addAttribute("module_alias", b.getStringAttr("_functools"));
  b.create(helperImport);
  unsigned index = 0;
  const auto prepared = Clock::now();
  Nanoseconds parseTime{}, lowerTime{};
  if (failed(splitAndProcessBuffer(
      std::move(buffer),
      [&](std::unique_ptr<llvm::MemoryBuffer> chunk, llvm::raw_ostream &) {
        llvm::SourceMgr source;
        source.AddNewSourceBuffer(std::move(chunk), llvm::SMLoc());
        const auto begin = Clock::now();
        auto input = parseSourceFile<ModuleOp>(source, &context);
        const auto parsed = Clock::now();
        auto result = input ? lower(*input, *output, index++) : failure();
        parseTime += std::chrono::duration_cast<Nanoseconds>(parsed - begin);
        lowerTime += std::chrono::duration_cast<Nanoseconds>(Clock::now() - parsed);
        return result;
      },
      llvm::nulls())))
    return 1;
  const auto checking = Clock::now();
  if (failed(verify(*output)))
    return 1;
  const auto emitting = Clock::now();
  if (failed(py::translateToPython(output->getOperation(), destination)))
    return 1;
  destination.flush();
  if (std::getenv("RIDGER_EMITPY_PROFILE")) {
    llvm::errs() << "graphs=" << index
                 << " setup_ns=" << std::chrono::duration_cast<Nanoseconds>(prepared - started).count()
                 << " parse_verify_ns=" << parseTime.count()
                 << " conversion_ns=" << lowerTime.count()
                 << " emitpy_verify_ns=" << std::chrono::duration_cast<Nanoseconds>(emitting - checking).count()
                 << " translation_ns=" << std::chrono::duration_cast<Nanoseconds>(Clock::now() - emitting).count()
                 << '\n';
  }
  return 0;
}

constexpr unsigned MaxFrame = 4 * 1024 * 1024;

void reply(char status, StringRef payload) {
  const auto size = payload.size();
  const char header[] = {status, char(size >> 24), char(size >> 16),
                         char(size >> 8), char(size)};
  std::cout.write(header, sizeof(header));
  std::cout.write(payload.data(), size);
  std::cout.flush();
}

int serve(MLIRContext &context) {
  std::cout.write("EP01", 4);
  std::cout.flush();
  for (;;) {
    unsigned char header[4];
    if (!std::cin.read(reinterpret_cast<char *>(header), sizeof(header)))
      return std::cin.eof() && std::cin.gcount() == 0 ? 0 : 1;
    unsigned size = (unsigned(header[0]) << 24) | (unsigned(header[1]) << 16) |
                    (unsigned(header[2]) << 8) | unsigned(header[3]);
    if (size == 0)
      return 0;
    if (size > MaxFrame)
      return 1;
    std::string input(size, '\0');
    if (!std::cin.read(input.data(), size))
      return 1;
    if (input.find('\0') != std::string::npos) {
      reply('E', "embedded NUL in MLIR");
      continue;
    }
    std::string output;
    llvm::raw_string_ostream source(output);
    if (translate(context, llvm::MemoryBuffer::getMemBufferCopy(input, "request"),
                  source) != 0) {
      reply('E', "MLIR translation failed");
    } else if (output.size() > MaxFrame) {
      reply('E', "Python source exceeds frame limit");
    } else {
      reply('O', output);
    }
    if (!std::cout)
      return 1;
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << "usage: ridger-emitpy IRDL INPUT | IRDL --serve | --literal STRING\n";
    return 2;
  }
  DialectRegistry registry;
  registry.insert<func::FuncDialect, irdl::IRDLDialect, py::EmitPyDialect>();
  MLIRContext context(registry);
  context.disableMultithreading();
  context.getOrLoadDialect<py::EmitPyDialect>();
  if (StringRef(argv[1]) == "--literal") {
    OpBuilder b(&context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<ModuleOp> output(ModuleOp::create(loc));
    b.setInsertionPointToStart(output->getBody());
    Lowering adapter{b, loc, py::OpaqueType::get(&context, "object")};
    adapter.path({argv[2]});
    return failed(verify(*output)) ||
           failed(py::translateToPython(output->getOperation(), llvm::outs()));
  }

  auto definitions = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!definitions || failed(irdl::loadDialects(*definitions)))
    return 1;
  if (StringRef(argv[2]) == "--serve")
    return serve(context);
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(argv[2]);
  if (!buffer) {
    llvm::errs() << buffer.getError().message() << '\n';
    return 1;
  }

  return translate(context, std::move(*buffer), llvm::outs());
}
