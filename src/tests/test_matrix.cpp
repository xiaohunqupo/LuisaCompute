#include <luisa/core/mathematics.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/logging.h>
#include <random>
using namespace luisa;

void print_matrix_in_row(float4x4 matrix) {
    luisa::string result;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result += luisa::format("{}, ", matrix.cols[col][row]);
        }
        result += "\n";
    }
    LUISA_INFO("Print Matrix in row-major:\n{}", result);
}
void print_matrix_in_col(float4x4 matrix) {
    luisa::string result;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            result += luisa::format("{}, ", matrix.cols[col][row]);
        }
        result += "\n";
    }
    LUISA_INFO("Print Matrix in column-major:\n{}", result);
}

float4 get_matrix_row(float4x4 matrix, uint32_t row_index) {
    return float4(matrix.cols[0][row_index], matrix.cols[1][row_index], matrix.cols[2][row_index], matrix.cols[3][row_index]);
}

int main() {
    //////////////////////////////////// Layout
    // Luisa's matrix is column-based matrix, for instance, the memory layout of float4x4 is:
    float4x4 my_matrix = make_float4x4(
        float4(11, 12, 13, 14),//  colume 0
        float4(21, 22, 23, 24),//  colume 1
        float4(31, 32, 33, 34),//  colume 1
        float4(41, 42, 43, 44) //  colume 1
    );
    print_matrix_in_row(my_matrix);
    print_matrix_in_col(my_matrix);
    /*
    print's in row-order, should be transposed:
        11, 21, 31, 41,  
        12, 22, 32, 42,  
        13, 23, 33, 43, 
        14, 24, 34, 44,
    */

    //////////////////////////////////// Matrix-Matrix multiplication
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    float4x4 lhs_matrix;
    float4x4 rhs_matrix;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            lhs_matrix.cols[col][row] = dist(gen);
            rhs_matrix.cols[col][row] = dist(gen);
        }
    }
    float4x4 result = lhs_matrix * rhs_matrix;
    // manually multiply
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // Result[col][row] =  LHS[row] * RHS[col]
            float multiply_result = dot(get_matrix_row(lhs_matrix, row), rhs_matrix.cols[col]);
            // check if result is equal
            LUISA_ASSERT(abs(result.cols[col][row] - multiply_result) <= std::numeric_limits<float>::epsilon(), "Test matrix-matrix multiply.");
        }
    }
    //////////////////////////////////// Matrix-Vector multiplication
    float4 rhs_vector{
        dist(gen),
        dist(gen),
        dist(gen),
        dist(gen)};
    float4 result_vector = lhs_matrix * rhs_vector;
    for (int row = 0; row < 4; ++row) {
        float multiply_result = dot(get_matrix_row(lhs_matrix, row), rhs_vector);
        LUISA_ASSERT(abs(result_vector[row] - multiply_result) <= std::numeric_limits<float>::epsilon(), "Test matrix-vector  multiply.");
    }
}