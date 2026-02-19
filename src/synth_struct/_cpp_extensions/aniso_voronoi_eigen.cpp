// /synth_struct/src/synth_struct/_cpp_extensions/aniso_voronoi_eigen.cpp

// This is a C++ implementation of an anisotropic voronoi assignment
// This code uses the Eigen library to quickly calculate Voronoi
// assignments.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <limits>

namespace py = pybind11;

// Compute anisotropic distance using Eigen (vectorized)
template<int Dim>
inline float compute_aniso_distance_eigen(
    const Eigen::Matrix<float, Dim, 1>& point,
    const Eigen::Matrix<float, Dim, 1>& seed,
    const Eigen::Matrix<float, Dim, 1>& scale,
    const Eigen::Matrix<float, Dim, Dim, Eigen::RowMajor>& rotation
) {
    // diff = point - seed
    Eigen::Matrix<float, Dim, 1> diff = point - seed;
    
    // diff_rotated = R^T * diff (using transpose())
    Eigen::Matrix<float, Dim, 1> diff_rotated = rotation.transpose() * diff;
    
    // diff_scaled = diff_rotated / scale (element-wise division)
    Eigen::Matrix<float, Dim, 1> diff_scaled = diff_rotated.cwiseQuotient(scale);
    
    // Return squared norm
    return diff_scaled.squaredNorm();
}

// 2D specialization
py::array_t<int32_t> aniso_voronoi_2d(
    const Eigen::VectorXi& dimensions,
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& seeds,
    const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>& scale_factors,
    const std::vector<Eigen::Matrix<float, 2, 2, Eigen::RowMajor>>& rotations,
    int chunk_size
) {
    size_t total_voxels = dimensions[0] * dimensions[1];
    int num_grains = seeds.rows();
    
    py::array_t<int32_t> grain_ids_flat(total_voxels);
    auto result_buf = grain_ids_flat.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
    
    std::cout << "  Performing anisotropic Voronoi tessellation (C++ Eigen 2D)..." << std::endl;
    
    size_t chunk_count = 0;
    for (size_t start = 0; start < total_voxels; start += chunk_size) {
        size_t end = std::min(start + chunk_size, total_voxels);
        size_t chunk_len = end - start;
        
        // Preallocate chunk coordinates matrix
        Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> chunk_coords(chunk_len, 2);
        
        // Convert linear indices to coordinates
        for (size_t idx = 0; idx < chunk_len; ++idx) {
            size_t global_idx = start + idx;
            chunk_coords(idx, 1) = static_cast<float>(global_idx % dimensions[1]);
            chunk_coords(idx, 0) = static_cast<float>(global_idx / dimensions[1]);
        }
        
        // Process each point in parallel
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < chunk_len; ++p) {
            float min_distance = std::numeric_limits<float>::max();
            int32_t best_grain = 1;
            
            Eigen::Matrix<float, 2, 1> point = chunk_coords.row(p).transpose();
            
            for (int g = 0; g < num_grains; ++g) {
                Eigen::Matrix<float, 2, 1> seed = seeds.row(g).transpose();
                Eigen::Matrix<float, 2, 1> scale = scale_factors.row(g).transpose();
                
                float distance = compute_aniso_distance_eigen<2>(
                    point, seed, scale, rotations[g]
                );
                
                if (distance < min_distance) {
                    min_distance = distance;
                    best_grain = g + 1;
                }
            }
            
            result_ptr[start + p] = best_grain;
        }
        
        if (chunk_count % 10 == 0) {
            float progress = 100.0f * end / total_voxels;
            std::cout << "  Progress: " << progress << "%" << std::endl;
        }
        chunk_count++;
    }
    
    std::cout << "Done!" << std::endl;
    return grain_ids_flat;
}

// 3D specialization
py::array_t<int32_t> aniso_voronoi_3d(
    const Eigen::VectorXi& dimensions,
    const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& seeds,
    const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& scale_factors,
    const std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>& rotations,
    int chunk_size
) {
    size_t total_voxels = dimensions[0] * dimensions[1] * dimensions[2];
    int num_grains = seeds.rows();
    
    py::array_t<int32_t> grain_ids_flat(total_voxels);
    auto result_buf = grain_ids_flat.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
    
    std::cout << "  Performing anisotropic Voronoi tessellation (C++ Eigen 3D)..." << std::endl;
    
    size_t chunk_count = 0;
    for (size_t start = 0; start < total_voxels; start += chunk_size) {
        size_t end = std::min(start + chunk_size, total_voxels);
        size_t chunk_len = end - start;
        
        // Preallocate chunk coordinates matrix
        Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> chunk_coords(chunk_len, 3);
        
        // Convert linear indices to coordinates
        size_t dim1_dim2 = dimensions[1] * dimensions[2];
        for (size_t idx = 0; idx < chunk_len; ++idx) {
            size_t global_idx = start + idx;
            chunk_coords(idx, 2) = static_cast<float>(global_idx % dimensions[2]);
            size_t temp = global_idx / dimensions[2];
            chunk_coords(idx, 1) = static_cast<float>(temp % dimensions[1]);
            chunk_coords(idx, 0) = static_cast<float>(temp / dimensions[1]);
        }
        
        // Process each point in parallel
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < chunk_len; ++p) {
            float min_distance = std::numeric_limits<float>::max();
            int32_t best_grain = 1;
            
            Eigen::Matrix<float, 3, 1> point = chunk_coords.row(p).transpose();
            
            for (int g = 0; g < num_grains; ++g) {
                Eigen::Matrix<float, 3, 1> seed = seeds.row(g).transpose();
                Eigen::Matrix<float, 3, 1> scale = scale_factors.row(g).transpose();
                
                float distance = compute_aniso_distance_eigen<3>(
                    point, seed, scale, rotations[g]
                );
                
                if (distance < min_distance) {
                    min_distance = distance;
                    best_grain = g + 1;
                }
            }
            
            result_ptr[start + p] = best_grain;
        }
        
        if (chunk_count % 10 == 0) {
            float progress = 100.0f * end / total_voxels;
            std::cout << "  Progress: " << progress << "%" << std::endl;
        }
        chunk_count++;
    }
    
    std::cout << "Done!" << std::endl;
    return grain_ids_flat;
}

// Main dispatcher function
py::array_t<int32_t> aniso_voronoi_assignment_eigen(
    py::array_t<int32_t> dimensions,
    py::array_t<float> seeds,
    py::array_t<float> scale_factors,
    py::list rotations,
    int chunk_size = 500000
) {
    auto dims_buf = dimensions.request();
    auto seeds_buf = seeds.request();
    
    int ndim = dims_buf.shape[0];
    int num_grains = seeds_buf.shape[0];
    
    // Map NumPy arrays to Eigen
    Eigen::Map<Eigen::VectorXi> dims(
        static_cast<int*>(dims_buf.ptr), ndim
    );
    
    if (ndim == 2) {
        // 2D case
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>> seeds_eigen(
            static_cast<float*>(seeds_buf.ptr), num_grains, 2
        );
        
        auto scales_buf = scale_factors.request();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>> scales_eigen(
            static_cast<float*>(scales_buf.ptr), num_grains, 2
        );
        
        // Convert rotation matrices
        std::vector<Eigen::Matrix<float, 2, 2, Eigen::RowMajor>> rotations_eigen;
        rotations_eigen.reserve(num_grains);
        for (auto rot : rotations) {
            py::array_t<float> rot_array = py::cast<py::array_t<float>>(rot);
            auto rot_buf = rot_array.request();
            Eigen::Map<Eigen::Matrix<float, 2, 2, Eigen::RowMajor>> rot_eigen(
                static_cast<float*>(rot_buf.ptr)
            );
            rotations_eigen.push_back(rot_eigen);
        }
        
        return aniso_voronoi_2d(dims, seeds_eigen, scales_eigen, rotations_eigen, chunk_size);
        
    } else if (ndim == 3) {
        // 3D case
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> seeds_eigen(
            static_cast<float*>(seeds_buf.ptr), num_grains, 3
        );
        
        auto scales_buf = scale_factors.request();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> scales_eigen(
            static_cast<float*>(scales_buf.ptr), num_grains, 3
        );
        
        // Convert rotation matrices
        std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> rotations_eigen;
        rotations_eigen.reserve(num_grains);
        for (auto rot : rotations) {
            py::array_t<float> rot_array = py::cast<py::array_t<float>>(rot);
            auto rot_buf = rot_array.request();
            Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> rot_eigen(
                static_cast<float*>(rot_buf.ptr)
            );
            rotations_eigen.push_back(rot_eigen);
        }
        
        return aniso_voronoi_3d(dims, seeds_eigen, scales_eigen, rotations_eigen, chunk_size);
        
    } else {
        throw std::runtime_error("Only 2D and 3D are supported");
    }
}

PYBIND11_MODULE(aniso_voronoi_eigen, m) {
    m.doc() = "Anisotropic Voronoi tessellation using Eigen";
    m.def("aniso_voronoi_assignment", &aniso_voronoi_assignment_eigen,
          "Perform anisotropic Voronoi assignment with Eigen",
          py::arg("dimensions"),
          py::arg("seeds"),
          py::arg("scale_factors"),
          py::arg("rotations"),
          py::arg("chunk_size") = 500000);
}
