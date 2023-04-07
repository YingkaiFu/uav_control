import struct
pattern = '<8b'
s = struct.Struct(pattern)

out = s.unpack(b'\00\x80\x80\x80\00\x80\x10\x7f')
print(out)