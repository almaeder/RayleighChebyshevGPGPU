#include <gtest/gtest.h>
#include "diag_operator.h"

class matrix : public ::testing::Test {
protected:
    // Common inputs
    int matrix_size;
    int number_of_vectors;
    std::vector<CPX> data;

    // SetUp is run before each test
    void SetUp() override {
        // Initialize if necessary
        matrix_size = 5;
        number_of_vectors = 3;
        data.resize(matrix_size * number_of_vectors);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < matrix_size * number_of_vectors; i++) {
            double real_part = dis(gen);
            double imag_part = dis(gen);
            data[i] = CPX(real_part, imag_part);
        }
    }

    // TearDown is run after each test
    void TearDown() override {
        // Clean up if necessary
    }
};

TEST_F(
    matrix,
    initialize_operator_empty
){

    diag_operator<CPX, RC_host_matrix<CPX>> rc_operator;

    EXPECT_TRUE(true);
}

TEST_F(
    matrix,
    initialize_operator
){

    diag_operator<CPX, RC_host_matrix<CPX>> rc_operator(matrix_size);

    for (int i = 0; i < matrix_size; i++) {
        EXPECT_EQ(rc_operator.data[i], std::complex<double>(i + 1, i + 1));
    }
}

TEST_F(
    matrix,
    apply_operator_matrix
){

    diag_operator<CPX, RC_host_matrix<CPX>> rc_operator(matrix_size);

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    rc_operator.apply(rc_matrix);

    for (int j = 0; j <number_of_vectors; j++) {
        for (int i = 0; i < matrix_size; i++) {
            EXPECT_EQ(rc_matrix.mData[i + j * matrix_size], std::complex<double>(i + 1, i + 1) * data[i + j * matrix_size]);
        }
    }
}


TEST_F(
    matrix,
    apply_operator_matrix_alpha_beta
){

    diag_operator<CPX, RC_host_matrix<CPX>> rc_operator(matrix_size);

    RC_host_matrix<CPX> rc_matrix_in(matrix_size, number_of_vectors, data);
    RC_host_matrix<CPX> rc_matrix_out(matrix_size, number_of_vectors, data);

    CPX alpha = std::complex<double>(2.0, 2.0);
    CPX beta = std::complex<double>(3.0, 3.0);

    rc_operator.apply(rc_matrix_in, rc_matrix_out, alpha, beta);

    for (int j = 0; j <number_of_vectors; j++) {
        for (int i = 0; i < matrix_size; i++) {
            EXPECT_EQ(rc_matrix_out.mData[i + j * matrix_size], alpha * std::complex<double>(i + 1, i + 1) * data[i + j * matrix_size] + beta * data[i + j * matrix_size]);
        }
    }
}
