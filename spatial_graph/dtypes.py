import numpy as np


class DType:
    def __init__(self, dtype_str):
        self.as_string = dtype_str
        self.is_array = self.__is_array(dtype_str)

        if self.is_array:
            self.base, self.size = self.__parse_array_dtype(dtype_str)
            self.shape = (self.size,)
        else:
            self.base = dtype_str
            self.size = None
            self.shape = ()

    def __is_array(self, dtype):
        if "[" in dtype:
            if "]" not in dtype:
                raise RuntimeError(f"invalid array(?) dtype {dtype}")
            return True
        return False

    def __parse_array_dtype(self, dtype):
        dtype, size = dtype.split("[")
        size = int(size.split("]")[0])

        return dtype, size

    def to_pyxtype(self, use_memory_view=False, as_arrays=False):
        """Convert this dtype to the equivalent C/C++/PYX types.

        Args:

            use_memory_view:

                If set, will produce "[::1]" instead of "[dim]" for array types.

            as_arrays:

                Create arrays of the types, e.g., "int32_t[::1]" instead of
                "int32_t" for dtype "int32".
        """

        # is this an array type?
        if self.is_array:
            if as_arrays:
                suffix = "[:, ::1]"
            else:
                if use_memory_view:
                    suffix = "[::1]"
                else:
                    suffix = f"[{self.size}]"
        else:
            suffix = "" if not as_arrays else "[::1]"

        if self.base == "float32" or self.base == "float":
            dtype = "float"
        elif self.base == "float64" or self.base == "double":
            dtype = "double"
        else:
            # this might not work for all of them, this is just a fallback
            dtype = np.dtype(self.base).name + "_t"

        return dtype + suffix


def dtypes_to_struct(struct_name, dtypes):
    pyx_code = f"cdef struct {struct_name}:\n"
    for name, dtype in dtypes.items():
        pyx_code += f"    {dtype.to_pyxtype()} {name}\n"
    return pyx_code


def dtypes_to_arguments(dtypes, as_arrays=False):
    return ", ".join(
        [
            f"{dtype.to_pyxtype(use_memory_view=True, as_arrays=as_arrays)} " f"{name}"
            for name, dtype in dtypes.items()
        ]
    )


def dtypes_to_array_pointers(
    dtypes, indent, definition_only=False, assignment_only=False, array_index=None
):
    pyx_code = ""

    for name, dtype in dtypes.items():
        if dtype.is_array:
            pyx_code = "    " * indent
            if not assignment_only:
                pyx_code += f"cdef {dtype.to_pyxtype()} "
            pyx_code += f"_p_{name}"
            if not definition_only:
                if array_index:
                    pyx_code += f" = &{name}[{array_index}, 0]\n"
                else:
                    pyx_code += f" = &{name}[0]\n"
            else:
                pyx_code += "\n"

    return pyx_code


def dtypes_to_array_pointer_names(dtypes, array_index=None):
    return ", ".join(
        [
            f"_p_{name}"
            if dtype.is_array
            else (f"{name}[{array_index}]" if array_index else name)
            for name, dtype in dtypes.items()
        ]
    )
