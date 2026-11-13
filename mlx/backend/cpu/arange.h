// Copyright © 2023 Apple Inc.

#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

#include "mlx/array.h"
#include "mlx/backend/cpu/encoder.h"

namespace mlx::core {

namespace {

// Float-to-integer conversions outside the target range are undefined in C++.
// Keep the result at the corresponding limit instead of depending on a CPU
// instruction's out-of-range behavior.
template <typename T>
T arange_cast(double value) {
  if constexpr (std::is_integral_v<T>) {
    const double upper = std::ldexp(1.0, std::numeric_limits<T>::digits);
    const double lower = std::is_signed_v<T> ? -upper : 0.0;
    if (value >= upper) {
      return std::numeric_limits<T>::max();
    }
    if (value <= lower) {
      return std::numeric_limits<T>::lowest();
    }
    if (std::isnan(value)) {
      return 0;
    }
  }
  return static_cast<T>(value);
}

template <typename T>
void arange(T start, T next, array& out, size_t size, Stream stream) {
  auto ptr = out.data<T>();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([ptr, start, next, size]() mutable {
    if constexpr (is_floating_point_v<T>) {
      auto step_size = next - start;
      // Preserve the typed step used on the GPU, but avoid repeated
      // additions in the output dtype, which accumulate rounding errors.
      double first = static_cast<double>(start);
      double step = static_cast<double>(step_size);
      // A zero index must return start even if the typed step is infinite.
      if (size > 0) {
        ptr[0] = start;
      }
      for (size_t i = 1; i < size; ++i) {
        double offset = step * static_cast<double>(i);
        ptr[i] = static_cast<T>(first + offset);
      }
    } else if constexpr (std::is_integral_v<T>) {
      // Integer ranges also need to wrap when a step overflows the dtype.
      using U = std::make_unsigned_t<T>;
      U value = static_cast<U>(start);
      U increment = static_cast<U>(next) - value;
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<T>(value);
        value += increment;
      }
    } else {
      auto step_size = next - start;
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = start;
        start += step_size;
      }
    }
  });
}

} // namespace

} // namespace mlx::core
