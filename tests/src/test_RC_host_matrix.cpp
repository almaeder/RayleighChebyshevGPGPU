#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "../../RC_host_matrix.h"

using CPX = std::complex<double>;

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
    initialize_matrix_empty
){

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors);

    EXPECT_EQ(rc_matrix.get_size(), matrix_size * number_of_vectors);
    EXPECT_EQ(rc_matrix.get_row_size(), matrix_size);
    EXPECT_EQ(rc_matrix.get_col_size(), number_of_vectors);
}

TEST_F(
    matrix,
    initialize_matrix_vector
){

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    for (int i = 0; i < matrix_size * number_of_vectors; i++) {
        EXPECT_EQ(rc_matrix.mData[i], data[i]);
    }

    EXPECT_EQ(rc_matrix.get_row_size(), matrix_size);
    EXPECT_EQ(rc_matrix.get_col_size(), number_of_vectors);
}

TEST_F(
    matrix,
    initialize_matrix_matrix
){

    RC_host_matrix<CPX> rc_matrix_ref(matrix_size, number_of_vectors, data);

    RC_host_matrix<CPX> rc_matrix(rc_matrix_ref);

    for (int i = 0; i < matrix_size * number_of_vectors; i++) {
        EXPECT_EQ(rc_matrix.mData[i], rc_matrix_ref.mData[i]);
    }

    EXPECT_EQ(rc_matrix.get_row_size(), matrix_size);
    EXPECT_EQ(rc_matrix.get_col_size(), number_of_vectors);
}
