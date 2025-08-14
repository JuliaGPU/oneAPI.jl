"""
    ze_make_version(major::Integer, minor::Integer) -> UInt32

32-bit unsigned integer version number from major and minor components.
This should be the Julia equivalent of the C macro:
`#define ZE_MAKE_VERSION( _major, _minor )  (( _major << 16 )|( _minor & 0x0000ffff))`
"""
function ZE_MAKE_VERSION(major::Integer, minor::Integer)
    # Shift the major version 16 bits to the left
    # and combine it with the minor version using a bitwise OR.
    # The `& 0xffff` is implicit for standard integer types when combining,
    # but we can be explicit if needed. The result is cast to UInt32.
    return (UInt32(major) << 16) | (UInt32(minor) & 0x0000ffff)
end
unmake_version(ver) = VersionNumber(Int(ver) >> 16, Int(ver) & 0x0000ffff)
