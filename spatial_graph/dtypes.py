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

    @property
    def base_c_type(self):
        """Convert the base of this DType into the equivalent C/C++ type."""

        if self.base == "float32" or self.base == "float":
            return "float"
        elif self.base == "float64" or self.base == "double":
            return "double"
        else:
            # this might not work for all of them, this is just a fallback
            return np.dtype(self.base).name + "_t"

    def to_c_decl(self, name):
        """Convert this dtype to the equivalent C/C++ declaration with the
        given name:

            "base_c_type name"        if not an array
            "base_c_type name[size]"  if an array type
        """
        # is this an array type?
        if self.is_array:
            suffix = f"[{self.size}]"
        else:
            suffix = ""

        return self.base_c_type + " " + name + suffix

    def to_pyxtype(self, use_memory_view=False, add_dim=False):
        """Convert this dtype to the equivalent PYX type.

            "base_c_type"
            "base_c_type[size]"     if an array type
            "base_c_type[::1]"      if an array type and use_memory_view
            "base_c_type[::1]"      if not an array type and add_dim
            "base_c_type[:, ::1]"   if an array type and add_dim

        Args:

            use_memory_view:

                If set, will produce "dtype[::1]" instead of "dtype[dim]" for
                array types.

            add_dim:

                Append a dim to the type, e.g., "int32_t[::1]" instead of
                "int32_t" for dtype "int32". If this DType is already an array,
                will create a 2D array, e.g., "int32_t[:, ::1]".
        """

        # is this an array type?
        if self.is_array:
            if add_dim:
                suffix = "[:, ::1]"
            else:
                if use_memory_view:
                    suffix = "[::1]"
                else:
                    suffix = f"[{self.size}]"
        else:
            suffix = "[::1]" if add_dim else ""

        return self.base_c_type + suffix

    def to_rvalue(self, name, array_index=None):
        """Convert this dtype into an r-value to be used in PYX files for
        assignments.

            "name"                  default
            "name[array_index]"     if array_index is given
            "{name[0], ..., name[size-1]}"
                                    if an array type
            "{name[array_index, 0], ..., name[array_index, size-1]}"
                                    if an array type and array_index is given
        """

        if self.is_array:
            if array_index:
                return (
                    "{"
                    + ", ".join(
                        [name + f"[{array_index}, {i}]" for i in range(self.size)]
                    )
                    + "}"
                )
            else:
                return (
                    "{" + ", ".join([name + f"[{i}]" for i in range(self.size)]) + "}"
                )
        else:
            if array_index:
                return f"{name}[{array_index}]"
            else:
                return name


def dtypes_to_struct(struct_name, dtypes):
    pyx_code = f"cdef struct {struct_name}:\n"
    for name, dtype in dtypes.items():
        pyx_code += f"    {dtype.to_pyxtype()} {name}\n"
    return pyx_code
