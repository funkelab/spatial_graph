from __future__ import annotations

import re

# Define valid base types
VALID_BASE_TYPES = {
    # Floating-point types (no _t suffix)
    "float": "float",
    "float32": "float",
    "double": "double",
    "float64": "double",
    # Fixed-width integer types
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    # Platform-dependent types mapped to fixed-width equivalents
    "int": "int64_t",
    "uint": "uint64_t",
}

# Regex pattern for dtype validation and extraction
DTYPE_PATTERN = r"^({})(?:\[(\d+)\])?$".format("|".join(VALID_BASE_TYPES))
DTYPE_REGEX = re.compile(DTYPE_PATTERN)


class DType:
    """A class to represent a data type in C/C++ and Cython/PYX files.

    Parameters
    ----------
    dtype_str : str
        The data type string in the format "base_type[size]". The base_type must
        be one of the valid base types defined in VALID_BASE_TYPES, and size is
        optional.
    """

    def __init__(self, dtype_str: str) -> None:
        self.as_string = dtype_str
        self.base, self.size = self.__parse_array_dtype(dtype_str)
        self.is_array = self.size is not None
        self.shape = (self.size,) if self.is_array else ()

    def __parse_array_dtype(self, dtype_str: str) -> tuple[str, int | None]:
        """Parse the array dtype string into base type and size."""

        if not (match := DTYPE_REGEX.match(dtype_str)):
            raise ValueError(
                f"Invalid dtype string: {dtype_str!r}. Must have base type of "
                f"{list(VALID_BASE_TYPES)!r} and optional size in square brackets."
            )

        base = match.group(1)
        size = int(match.group(2)) if match.group(2) else None

        if base not in VALID_BASE_TYPES:  # pragma: no cover
            raise ValueError(f"Invalid base type: {base}")

        return base, size

    @property
    def base_c_type(self) -> str:
        """Convert the base of this DType into the equivalent C/C++ type."""
        return VALID_BASE_TYPES[self.base]

    def to_c_decl(self, name: str) -> str:
        """Convert this dtype to the equivalent C/C++ declaration with the given name.

        "base_c_type name"        if not an array
        "base_c_type name[size]"  if an array type
        """
        if self.is_array:
            return f"{self.base_c_type} {name}[{self.size}]"
        else:
            return f"{self.base_c_type} {name}"

    def to_pyxtype(self, use_memory_view: bool = False, add_dim: bool = False) -> str:
        """Convert this dtype to the equivalent PYX type.

            "base_c_type"
            "base_c_type[size]"     if an array type
            "base_c_type[::1]"      if an array type and use_memory_view
            "base_c_type[::1]"      if not an array type and add_dim
            "base_c_type[:, ::1]"   if an array type and add_dim

        Parameters
        ----------
        use_memory_view : bool
            If set, will produce "dtype[::1]" instead of "dtype[dim]" for
            array types.
        add_dim : bool
            Append a dim to the type, e.g., "int32_t[::1]" instead of
            "int32_t" for dtype "int32". If this DType is already an array,
            will create a 2D array, e.g., "int32_t[:, ::1]".
        """
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

    def to_rvalue(self, name: str, array_index: str | None = None) -> str:
        """Convert this dtype into an r-value to be used in PYX files for assignments.

        "name"                  default
        "name[array_index]"     if array_index is given
        "{name[0], ..., name[size-1]}"
                                if an array type
        "{name[array_index, 0], ..., name[array_index, size-1]}"
                                if an array type and array_index is given
        """

        if self.size:
            if array_index:
                return (
                    "{"
                    + ", ".join(
                        [f"{name}[{array_index}, {i}]" for i in range(self.size)]
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
