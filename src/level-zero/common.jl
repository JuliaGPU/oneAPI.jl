#define ZE_MAKE_VERSION( _major, _minor )  (( _major << 16 )|( _minor & 0x0000ffff))
unmake_version(ver) = VersionNumber(Int(ver) >> 16, Int(ver) & 0x0000ffff)
