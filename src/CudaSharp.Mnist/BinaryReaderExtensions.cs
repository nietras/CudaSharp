using System;
using System.Buffers.Binary;
using System.IO;

namespace CudaSharp.Mnist;

static class BinaryReaderExtensions
{
    public static uint ReadUInt32BigEndian(this BinaryReader reader)
    {
        var value = reader.ReadUInt32();
        return BitConverter.IsLittleEndian ? BinaryPrimitives.ReverseEndianness(value) : value;
    }
}
