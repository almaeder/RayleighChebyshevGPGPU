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


TEST_F(
    matrix,
    elementwise_reading
){

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < number_of_vectors; j++) {
            EXPECT_EQ(rc_matrix(i, j), data[i + j * matrix_size]);
        }
    }

}

TEST_F(
    matrix,
    elementwise_writing
){

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < number_of_vectors; j++) {
            rc_matrix(i, j) = CPX(0.0, 0.0);
        }
    }
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < number_of_vectors; j++) {
            EXPECT_EQ(rc_matrix(i, j), CPX(0.0, 0.0));
        }
    }

}



TEST_F(
    matrix,
    inner_product
){

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    for (int i = 0; i < number_of_vectors; i++) {

        CPX inner_product = rc_matrix.inner_product(i, i);

        CPX expected_inner_product = CPX(0.0, 0.0);
        for (int j = 0; j < matrix_size; j++) {
            expected_inner_product += rc_matrix.mData[i * matrix_size + j] * std::conj(rc_matrix.mData[i * matrix_size + j]);
        }

        // compare real and complex parts separately
        ASSERT_DOUBLE_EQ(inner_product.real(), expected_inner_product.real());
        ASSERT_DOUBLE_EQ(inner_product.imag(), expected_inner_product.imag());
    }

    for (int i = 0; i < number_of_vectors - 1; i++) {
        CPX inner_product = rc_matrix.inner_product(i, i + 1);

        CPX expected_inner_product = CPX(0.0, 0.0);
        for (int j = 0; j < matrix_size; j++) {
            expected_inner_product += rc_matrix.mData[i * matrix_size + j] * std::conj(rc_matrix.mData[(i + 1) * matrix_size + j]);
        }

        // compare real and complex parts separately
        ASSERT_DOUBLE_EQ(inner_product.real(), expected_inner_product.real());
        ASSERT_DOUBLE_EQ(inner_product.imag(), expected_inner_product.imag());
    }

}



TEST_F(
    matrix,
    orthogonalize_matrix
){

    RC_host_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    rc_matrix.orthogonalize();

    for (int i = 0; i < number_of_vectors; i++) {

        CPX inner_product = rc_matrix.inner_product(i, i);

        // compare real and complex parts separately
        ASSERT_DOUBLE_EQ(inner_product.real(), 1);
        ASSERT_DOUBLE_EQ(inner_product.imag(), 0);
    }

    for (int i = 0; i < number_of_vectors; i++) {
        for (int j = 0; j < number_of_vectors; j++) {
            if (i == j) continue;

            CPX inner_product = rc_matrix.inner_product(i, j);

            // compare real and complex parts separately
            ASSERT_NEAR(inner_product.real(), 0, 1e-15);
            ASSERT_NEAR(inner_product.imag(), 0, 1e-15);
        }
    }


}
