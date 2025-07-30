#pragma once
#include <torch/torch.h>
#include <pybind11/numpy.h>
#include "astra_utils.h"

namespace py = pybind11;

namespace astra {

// Static assertions to ensure Action is compatible with int32 for no-copy conversion
static_assert(sizeof(Action) == sizeof(int32_t), "Action must be same size as int32_t for no-copy conversion");
static_assert(std::is_integral_v<Action>, "Action must be an integral type");

/* ---------------------------------------------------------------------------
 * numpy_of_tensor (templated version) ---------------------------------------
 * - Accepts a *CPU, contiguous* tensor of type T (const & to avoid ref-count
 *   churn and signal "read-only" intent).
 * - Returns a NumPy `py::array` that **shares the same memory**.
 * - Uses a capsule that owns a *copy of the intrusive_ptr* so the storage
 *   stays alive until Python drops its last reference.
 * - Supports float and Action (int) types.
 * -------------------------------------------------------------------------*/
template<typename T>
inline py::array numpy_of_tensor(const at::Tensor& t)
{
    if (!t.defined())       throw std::runtime_error("Tensor is undefined");
    if (!t.is_cpu())        throw std::runtime_error("Must be on CPU");
    if (!t.is_contiguous()) throw std::runtime_error("Must be contiguous");
    
    // Check scalar type matches T
    if constexpr (std::is_same_v<T, float>) {
        if (t.scalar_type() != at::kFloat) throw std::runtime_error("Expect float32");
    } else if constexpr (std::is_same_v<T, Action>) {
        if (t.scalar_type() != at::kInt) throw std::runtime_error("Expect int32");
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for tensor conversion. Only float and Action are supported.");
    }

    // Keep the tensor's storage alive: duplicate the intrusive_ptr in a capsule
    auto* holder = new at::Tensor(t);   // cheap (increments refcount)

    auto sizes   = t.sizes();
    auto strides = t.strides();         // element-strides
    ssize_t ndim = sizes.size();
    std::vector<ssize_t> shape(ndim), byte_strides(ndim);
    for (ssize_t i = 0; i < ndim; ++i) {
        shape[i]        = sizes[i];
        byte_strides[i] = strides[i] * sizeof(T);
    }

    return py::array(
        py::buffer_info(
            t.data_ptr<T>(),
            sizeof(T),
            py::format_descriptor<T>::format(),
            ndim,
            shape,
            byte_strides),
        py::capsule(holder, [](void* p){ delete reinterpret_cast<at::Tensor*>(p); })
    );
}

/* ---------------------------------------------------------------------------
 * tensor_of_numpy (templated version) ---------------------------------------
 * - Accepts a NumPy array (const & again).
 * - Builds a tensor with `torch::from_blob` → **zero-copy** view.
 * - The deleter captures a *py::object* copy of the array, so the array's
 *   refcount stays > 0 for as long as the tensor lives.
 * - Supports float and Action (int) types.
 * -------------------------------------------------------------------------*/
template<typename T>
inline at::Tensor tensor_of_numpy(const py::array& arr)
{
    py::buffer_info info = arr.request();               //  ➜ owns GIL
    if (info.itemsize != sizeof(T))
        throw std::runtime_error("Expected dtype with itemsize " + std::to_string(sizeof(T)));

    std::vector<int64_t> sizes(info.shape.begin(), info.shape.end());
    std::vector<int64_t> strides(info.ndim);
    for (int i = 0; i < info.ndim; ++i)
        strides[i] = static_cast<int64_t>(info.strides[i] / sizeof(T));

    // Capture a *new* reference so the Python array outlives the tensor
    py::object owner = arr;

    torch::ScalarType scalar_type;
    if constexpr (std::is_same_v<T, float>) {
        scalar_type = torch::kFloat32;
    } else if constexpr (std::is_same_v<T, Action>) {
        scalar_type = torch::kInt32;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for tensor conversion. Only float and Action are supported.");
    }

    return torch::from_blob(
        info.ptr,
        std::move(sizes),
        std::move(strides),
        [owner](void*) mutable { },     // called when tensor dies
        torch::dtype(scalar_type)
    );
}

/* ---------------------------------------------------------------------------
 * numpy_to_vector (templated helper) ----------------------------------------
 * - Converts a 1D NumPy array to std::vector<T>
 * - Used for converting action arrays and other 1D data
 * -------------------------------------------------------------------------*/
template<typename T>
inline std::vector<T> numpy_to_vector(const py::array_t<T>& array_np) {
    py::buffer_info buf = array_np.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Array must be 1-dimensional.");
    }
    
    T* ptr = static_cast<T*>(buf.ptr);
    size_t size = buf.shape[0];
    
    std::vector<T> result;
    result.reserve(size);
    result.assign(ptr, ptr + size);
    return result;
}

/* ---------------------------------------------------------------------------
 * vector_to_numpy (templated helper) ----------------------------------------
 * - Converts a std::vector<T> to a 1D NumPy array
 * - Creates a new NumPy array that owns its own copy of the data
 * - Supports float and Action (int) types
 * -------------------------------------------------------------------------*/
template<typename T>
inline py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    // Create a new numpy array with owned data
    auto result = py::array_t<T>(vec.size());
    py::buffer_info buf = result.request();
    
    T* ptr = static_cast<T*>(buf.ptr);
    
    // Copy data directly - simplified without GIL management
    std::copy(vec.begin(), vec.end(), ptr);
    
    return result;
}

/* ---------------------------------------------------------------------------
 * vector_to_numpy_nocopy (zero-copy version) --------------------------------
 * - Transfers ownership of vector's data to NumPy array (zero-copy)
 * - The vector is moved and its data is directly used by NumPy
 * - More efficient but requires careful lifetime management
 * -------------------------------------------------------------------------*/
template<typename T>
inline py::array_t<T> vector_to_numpy_nocopy(std::vector<T>&& vec) {
    // Move the vector to heap-allocated storage
    auto* heap_vec = new std::vector<T>(std::move(vec));
    
    // Create NumPy array that directly uses the vector's data
    auto result = py::array_t<T>(
        heap_vec->size(),                    // size
        heap_vec->data(),                    // data pointer
        py::capsule(heap_vec, [](void* p) { // custom deleter
            delete reinterpret_cast<std::vector<T>*>(p);
        })
    );
    
    return result;
}

}