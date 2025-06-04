#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/reg2mem.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void lower_phi_nodes_in_function(FunctionDefinition *def, Reg2MemInfo &info) noexcept {
    luisa::vector<PhiInst *> phi_nodes;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { phi_nodes.emplace_back(static_cast<PhiInst *>(inst)); }
    });
    for (auto phi : phi_nodes) {
        lower_phi_node_to_local_variable(phi);
    }
    info.lowered_phi_count += phi_nodes.size();
}

static void hoist_allocas_to_top_of_funtion(FunctionDefinition *def) noexcept {
    luisa::vector<AllocaInst *> allocas;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<AllocaInst>()) {
            allocas.emplace_back(static_cast<AllocaInst *>(inst));
        }
    });
    XIRBuilder b;
    b.set_insertion_point(def->body_block()->instructions().head_sentinel());
    for (auto a : allocas) { b.append(a->remove_self()); }
}

static void run_reg2mem_pass_on_function(Function *function, Reg2MemInfo &info) noexcept {
    if (auto definition = function->definition()) {
        lower_phi_nodes_in_function(definition, info);
        hoist_allocas_to_top_of_funtion(definition);
    }
}

}// namespace detail

Reg2MemInfo reg2mem_pass_run_on_function(Function *function) noexcept {
    Reg2MemInfo info;
    detail::run_reg2mem_pass_on_function(function, info);
    return info;
}

Reg2MemInfo reg2mem_pass_run_on_module(Module *module) noexcept {
    Reg2MemInfo info;
    for (auto f : module->function_list()) {
        detail::run_reg2mem_pass_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
