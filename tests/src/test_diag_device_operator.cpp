#include <gtest/gtest.h>
#include "diag_device_operator.h"
#include "../../RayleighChebyshevLMCuda.h"

class device_matrix : public ::testing::Test
{
protected:
    // Common inputs
    int matrix_size;
    int number_of_vectors;
    std::vector<CPX> data;

    // SetUp is run before each test
    void SetUp() override
    {
        // Initialize if necessary
        matrix_size = 5;
        number_of_vectors = 3;
        data.resize(matrix_size * number_of_vectors);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < matrix_size * number_of_vectors; i++)
        {
            double real_part = dis(gen);
            double imag_part = dis(gen);
            data[i] = CPX(real_part, imag_part);
        }
    }

    // TearDown is run after each test
    void TearDown() override
    {
        // Clean up if necessary
    }
};

TEST_F(
    device_matrix,
    initialize_operator_empty)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator;

    EXPECT_TRUE(true);
}

TEST_F(
    device_matrix,
    initialize_operator)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    for (int i = 0; i < matrix_size; i++)
    {
        EXPECT_EQ(rc_operator.data[i], std::complex<double>(i + 1, 0));
    }
}

TEST_F(
    device_matrix,
    matrix_initialization)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    RC_device_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    for (int j = 0; j < number_of_vectors; j++)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            EXPECT_EQ(rc_matrix(i,j), data[i + j * matrix_size]);
        }
    }
}

TEST_F(
    device_matrix,
    matrix_copy)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    RC_device_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    rc_matrix.device_to_host_copy();

    for (int j = 0; j < number_of_vectors; j++)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            EXPECT_EQ(rc_matrix(i,j), data[i + j * matrix_size]);
        }
    }
}

TEST_F(
    device_matrix,
    matrix_copy2)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    RC_device_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    rc_matrix.host_to_device_copy();
    rc_matrix.device_to_host_copy();

    for (int j = 0; j < number_of_vectors; j++)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            EXPECT_EQ(rc_matrix(i,j), data[i + j * matrix_size]);
        }
    }
}

TEST_F(
    device_matrix,
    apply_operator_matrix)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    RC_device_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    rc_operator.apply(rc_matrix);

    for (int j = 0; j < number_of_vectors; j++)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            ASSERT_NEAR(rc_matrix(i,j).real(), (std::complex<double>(i + 1, 0) * data[i + j * matrix_size]).real(), 1e-14);
            ASSERT_NEAR(rc_matrix(i,j).imag(), (std::complex<double>(i + 1, 0) * data[i + j * matrix_size]).imag(), 1e-14);
        }
    }
}

TEST_F(
    device_matrix,
    apply_operator_matrix_alpha_beta)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    RC_device_matrix<CPX> rc_matrix_in(matrix_size, number_of_vectors, data);
    RC_device_matrix<CPX> rc_matrix_out(matrix_size, number_of_vectors, data);

    CPX alpha = std::complex<double>(2.0, 2.0);
    CPX beta = std::complex<double>(3.0, 3.0);

    rc_matrix_in.host_to_device_copy();
    rc_matrix_out.host_to_device_copy();

    rc_operator.apply(rc_matrix_in, rc_matrix_out, alpha, beta);

    rc_matrix_out.device_to_host_copy();

    for (int j = 0; j < number_of_vectors; j++)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            ASSERT_NEAR(rc_matrix_out(i,j).real(), (alpha * std::complex<double>(i + 1, 0) * data[i + j * matrix_size] + beta * data[i + j * matrix_size]).real(), 1e-14);
            ASSERT_NEAR(rc_matrix_out(i,j).imag(), (alpha * std::complex<double>(i + 1, 0) * data[i + j * matrix_size] + beta * data[i + j * matrix_size]).imag(), 1e-14);
        }
    }
}

TEST_F(
    device_matrix,
    init_solver)
{

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, number_of_vectors);

    RC_device_matrix<CPX> rc_matrix(matrix_size, number_of_vectors, data);

    RC_device_randomize<CPX> rc_randomize;

    RayleighChebyshevLMCuda<
        RC_device_matrix<CPX>,
        RCvector<CPX>,
        diag_device_operator<CPX, RC_device_matrix<CPX>>,
        RC_device_randomize<CPX>,
        CPX>
        rc_procedure;

    std::string stop_condition = "RESIDUAL_ONLY";

    (rc_procedure).setStopCondition(stop_condition);
    (rc_procedure).setEigDiagnosticsFlag(true);
    (rc_procedure).setVerboseFlag(true);

    EXPECT_TRUE(true);
}

TEST_F(
    device_matrix,
    solve)
{


    RC_device_randomize<CPX> rc_randomize;

    RC_device_matrix<CPX> eig_vectors;


    RayleighChebyshevLMCuda<
        RC_device_matrix<CPX>,
        RCvector<CPX>,
        diag_device_operator<CPX, RC_device_matrix<CPX>>,
        RC_device_randomize<CPX>,
        CPX>
        rc_procedure;

    std::string stop_condition = "RESIDUAL_ONLY";

    (rc_procedure).setStopCondition(stop_condition);
    (rc_procedure).setEigDiagnosticsFlag(true);
    (rc_procedure).setVerboseFlag(true);

    int eig_count = 2;
    double subspace_tol = 1e-6;
    int subspace_size = 2;
    int buffer_size = 2;
    RCvector<CPX> vTmp(matrix_size);
    std::vector<double> eig_values;

    eig_vectors.resize(matrix_size, subspace_size + buffer_size);

    diag_device_operator<CPX, RC_device_matrix<CPX>> rc_operator(matrix_size, subspace_size + buffer_size);

    rc_procedure.getMinEigenSystem(
                    eig_count,
                    subspace_tol,
                    subspace_size,
                    buffer_size,
                    vTmp,
                    rc_operator,
                    rc_randomize,
                    eig_values,
                    eig_vectors);

    for(int i = 0; i < eig_count; i++)
    {
        // NOTE: Tol is for residual and not for eig value
        ASSERT_NEAR(eig_values[i], (i+1), subspace_tol);
    }

}
