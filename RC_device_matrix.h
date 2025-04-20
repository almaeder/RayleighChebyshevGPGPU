
// #pragma once
// #include "RC_Vector.h"
// #include "RC_Abstract_Matrix.h"

// #ifndef RC_HOST_MATRIX_
// #define RC_HOST_MATRIX_

// template <typename T>
// class RC_host_matrix : public RC_abstract_matrix<T>
// {
// 	public:

// //////////////////////////////////////////////////////////
// //  Required constructors
// //////////////////////////////////////////////////////////

// 	RC_host_matrix();

//     RC_host_matrix(const RC_host_matrix<T>& W);

// //////////////////////////////////////////////////////////
// // Constructor not required for RayleighChebyshve
// // but useful for creating test program
// //////////////////////////////////////////////////////////

//     RC_host_matrix(RC_INT m, RC_INT n);

// 	~RC_host_matrix();

// //////////////////////////////////////////////////////////////////
// //  Member functions required to use this class as a
// //  RayleighChebyshev template parameter
// //////////////////////////////////////////////////////////////////


// 	void host_to_device_copy();

// 	void device_to_host_copy();


// 	void initialize(const RC_host_matrix<T>& W);

// 	void initialize(const RCvector<T>& V, RC_INT n);

// 	void orthogonalize();

//     RC_host_matrix<T>& operator=(const RC_host_matrix<T>& W);

// 	void normalize();

// 	template <typename T1>
// 	T1 inner_product(const RC_INT k, const RC_INT l) const;


// 	std::complex<double> innerprod_complex(const RC_INT k, const RC_INT l) const;

// 	double innerprod_real(const RC_INT k, const RC_INT l) const;

// 	void resize_rows(RC_INT n);


// 	void resize_rows(RC_INT n, T value);
	

// 	void resize_rows(RC_INT n, RCvector<T>& V);

// 	void resize(RC_INT m, RC_INT n);

// 	std::vector<T> mData;

// 	RC_INT n;
// 	RC_INT m;
// };


// #endif /* RC_HOST_MATRIX_ */
