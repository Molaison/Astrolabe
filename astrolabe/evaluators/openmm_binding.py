"""Verify OpenCL and CUDA name the same allocated physical GPU.

OpenCL device indices do not inherit CUDA_VISIBLE_DEVICES. The comparison uses
PCI identity from the Khronos cl_khr_pci_bus_info query and an allocation-scoped
NVIDIA driver query. Merely specifying an index is not accepted as evidence.
"""
import ctypes
import ctypes.util
import re
import subprocess


def _index(value):
    text = str(value)
    if not text.isdecimal() or str(int(text)) != text:
        raise ValueError('OpenCL requires one explicit nonnegative device/platform index')
    return int(text)


def _opencl_device_pci(platform_index, device_index):
    library = ctypes.util.find_library('OpenCL')
    if not library:
        raise RuntimeError('cannot verify OpenCL binding: OpenCL loader unavailable')
    cl = ctypes.CDLL(library)
    uint, ulong, pointer = ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_void_p
    cl.clGetPlatformIDs.argtypes = [uint, ctypes.POINTER(pointer), ctypes.POINTER(uint)]
    cl.clGetPlatformIDs.restype = ctypes.c_int
    cl.clGetDeviceIDs.argtypes = [pointer, ulong, uint, ctypes.POINTER(pointer), ctypes.POINTER(uint)]
    cl.clGetDeviceIDs.restype = ctypes.c_int
    cl.clGetDeviceInfo.argtypes = [pointer, uint, ctypes.c_size_t, pointer, ctypes.POINTER(ctypes.c_size_t)]
    cl.clGetDeviceInfo.restype = ctypes.c_int
    def checked(status):
        if status:
            raise RuntimeError(f'cannot verify OpenCL physical binding (error {status})')
    count = uint()
    checked(cl.clGetPlatformIDs(0, None, ctypes.byref(count)))
    if platform_index >= count.value:
        raise ValueError('OpenCL platform index does not exist')
    platforms = (pointer * count.value)()
    checked(cl.clGetPlatformIDs(count.value, platforms, None))
    checked(cl.clGetDeviceIDs(platforms[platform_index], 0xFFFFFFFF, 0, None, ctypes.byref(count)))
    if device_index >= count.value:
        raise ValueError('OpenCL device index does not exist')
    devices = (pointer * count.value)()
    checked(cl.clGetDeviceIDs(platforms[platform_index], 0xFFFFFFFF, count.value, devices, None))
    device = devices[device_index]
    kind = ulong()
    checked(cl.clGetDeviceInfo(device, 0x1000, ctypes.sizeof(kind), ctypes.byref(kind), None))
    if kind.value & 2 and not kind.value & 4:
        return {'kind': 'cpu', 'pci': None}
    pci = (uint * 4)()
    checked(cl.clGetDeviceInfo(device, 0x410F, ctypes.sizeof(pci), ctypes.byref(pci), None))
    return {'kind': 'gpu', 'pci': tuple(int(value) for value in pci)}


def _cuda_device_pci(allocated_token):
    if not allocated_token or ',' in allocated_token or allocated_token.startswith('MIG-'):
        raise ValueError('OpenCL binding requires one full physical GPU allocation')
    result = subprocess.run(['nvidia-smi', '--id=' + allocated_token, '--query-gpu=uuid,pci.bus_id',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=5)
    rows = [row.strip().split(',') for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError('allocated GPU PCI identity was not unique')
    uuid, address = [value.strip() for value in rows[0]]
    match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)', address)
    if not match or not uuid.startswith('GPU-'):
        raise RuntimeError('driver did not report a complete allocated GPU identity')
    return {'uuid': uuid, 'pci': tuple(int(value, 16) for value in match.groups())}


def verify_opencl_binding(properties, allocated_token):
    if not {'OpenCLPlatformIndex', 'DeviceIndex'}.issubset(properties):
        raise ValueError('OpenCL requires explicit OpenCLPlatformIndex and DeviceIndex')
    platform_index, device_index = _index(properties['OpenCLPlatformIndex']), _index(properties['DeviceIndex'])
    observed = _opencl_device_pci(platform_index, device_index)
    if observed['kind'] == 'cpu':
        return {**observed, 'platform_index': platform_index, 'device_index': device_index}
    allocated = _cuda_device_pci(allocated_token)
    if observed['pci'] != allocated['pci']:
        raise ValueError('OpenCL relaxation device is outside the admitted physical GPU')
    return {**allocated, 'kind': 'gpu', 'platform_index': platform_index, 'device_index': device_index}
