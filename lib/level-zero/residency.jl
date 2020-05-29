export make_resident, evict


## memory

make_resident(dev::ZeDevice, buf::Buffer, size=sizeof(buf)) =
    zeDeviceEvictMemory(dev, buf, size)

evict(dev::ZeDevice, buf::Buffer, size=sizeof(buf)) =
    zeDeviceEvictMemory(dev, buf, size)
