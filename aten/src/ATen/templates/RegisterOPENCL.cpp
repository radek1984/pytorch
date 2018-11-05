#include <ATen/RegisterOPENCL.h>

// ${generated_comment}

#include <ATen/Type.h>
#include <ATen/Context.h>
#include <ATen/UndefinedType.h>
#include <ATen/core/VariableHooksInterface.h>

${opencl_type_headers}

namespace at {

void register_opencl_types(Context * context) {
  ${opencl_type_registrations}
}

} // namespace at
