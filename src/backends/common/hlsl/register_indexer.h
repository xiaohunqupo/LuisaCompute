#pragma once
#include "hlsl_codegen.h"
namespace lc::hlsl {
struct RegisterIndexer {
    virtual void init() = 0;
    virtual uint &get(uint idx) = 0;
};
struct DXILRegisterIndexer : public RegisterIndexer {
    std::array<uint, 3> values{};
    void init() override {
        values = {1, 0, 0};
    }
    uint &get(uint idx) override {
        return values[idx];
    }
};
struct SpirVRegisterIndexer : public RegisterIndexer {
    uint count{};
    void init() override {
        count = 0;
    }
    uint &get(uint idx) override {
        return count;
    }
};

}// namespace lc::hlsl