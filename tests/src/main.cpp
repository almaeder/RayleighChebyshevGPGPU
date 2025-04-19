#include <gtest/gtest.h>
#include <complex>

#define CPX std::complex<double>

class diag_matrix : public ::testing::Test {
protected:
    // Common inputs
    int nnz = 5;
    int matrix_size = 5;
    std::vector<int> col_indices;
    std::vector<int> row_indptr;
    std::vector<CPX> data;

    // SetUp is run before each test
    void SetUp() override {
        // Initialize if necessary

        col_indices.resize(5);
        row_indptr.resize(6);
        data.resize(5);

        nnz = 5;
        matrix_size = 5;
        for (int i = 0; i < 5; ++i) {
            col_indices[i] = i;
            row_indptr[i] = i;
            data[i] = CPX(i + 1, 0.0);
        }
        row_indptr[5] = 5;
    }

    // TearDown is run after each test
    void TearDown() override {
        // Clean up if necessary
    }
};


TEST_F(
    diag_matrix,
    initialize_empty
){
    EXPECT_TRUE(true);
}




// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}