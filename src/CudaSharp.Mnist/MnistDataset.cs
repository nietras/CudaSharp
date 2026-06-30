using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace CudaSharp.Mnist;

public static class MnistDataset
{
    const string UrlBase = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    const string TrainImagesFileName = "train-images-idx3-ubyte.gz";
    const string TrainLabelsFileName = "train-labels-idx1-ubyte.gz";
    const string TestImagesFileName = "train-images-idx3-ubyte.gz";
    const string TestLabelsFileName = "train-labels-idx1-ubyte.gz";

    static readonly IReadOnlyList<string> FileNames =
        [TrainImagesFileName, TrainLabelsFileName, TestImagesFileName, TestLabelsFileName];

    public static async Task EnsureDatasetFiles(string localDirectory)
    {
        foreach (var fileName in FileNames)
        {
            var fileUrl = UrlBase + fileName;
            var filePath = Path.Combine(localDirectory, fileName);
            await EnsureDatasetFile(fileUrl, filePath);
        }
    }

    public static async Task EnsureDatasetFile(string fileUrl, string filePath)
    {
        if (File.Exists(filePath)) { return; }

        Console.WriteLine($"[DOWNLOAD] MNIST dataset file missing. Fetching from: {fileUrl}");
        using var client = new HttpClient();
        var response = await client.GetAsync(fileUrl);
        response.EnsureSuccessStatusCode();

        using var fileStream = File.Create(filePath);
        await response.Content.CopyToAsync(fileStream);
        Console.WriteLine($"[DOWNLOAD] Download complete. Saved to: {filePath}");
    }


    public static Tensor<byte> ParseImagesGz(string filePath)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        Span<byte> header = stackalloc byte[4 * sizeof(uint)];
        gzStream.ReadExactly(header);

        var magic = BinaryPrimitives.ReadUInt32BigEndian(header.Slice(0 * sizeof(uint), sizeof(uint)));
        if (magic != 0x00000803)
            throw new InvalidOperationException($"Invalid images magic number: {magic:X}");

        var count = checked((int)BinaryPrimitives.ReadUInt32BigEndian(header.Slice(1 * sizeof(uint), sizeof(uint))));
        var rows = checked((int)BinaryPrimitives.ReadUInt32BigEndian(header.Slice(2 * sizeof(uint), sizeof(uint))));
        var cols = checked((int)BinaryPrimitives.ReadUInt32BigEndian(header.Slice(3 * sizeof(uint), sizeof(uint))));

        if (rows != 28 || cols != 28)
            throw new InvalidOperationException($"Expected 28x28 images, but got {rows}x{cols}");

        var images = Tensor.CreateFromShapeUninitialized<byte>([(nint)count, (nint)rows, (nint)cols]);
        var imageBytes = MemoryMarshal.CreateSpan(ref images.GetPinnableReference(), checked((int)images.FlattenedLength));
        gzStream.ReadExactly(imageBytes);
        return images;
    }

    public static Tensor<uint> PackImages(Tensor<byte> images)
    {
        var count = checked((int)images.Lengths[0]);
        var rows = checked((int)images.Lengths[1]);
        var cols = checked((int)images.Lengths[2]);
        var packedImages = Tensor.CreateFromShapeUninitialized<uint>([(nint)count, (nint)rows]);

        for (var i = 0; i < count; i++)
        {
            for (var r = 0; r < rows; r++)
            {
                uint rowBits = 0;
                for (var c = 0; c < cols; c++)
                {
                    if (images[i, r, c] > 127)
                    {
                        rowBits |= (1u << c);
                    }
                }
                packedImages[i, r] = rowBits;
            }
        }

        return packedImages;
    }

    public static Tensor<byte> ParseLabelsGz(string filePath)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        Span<byte> header = stackalloc byte[2 * sizeof(uint)];
        gzStream.ReadExactly(header);

        var magic = BinaryPrimitives.ReadUInt32BigEndian(header.Slice(0 * sizeof(uint), sizeof(uint)));
        if (magic != 0x00000801)
            throw new InvalidOperationException($"Invalid images magic number: {magic:X}");

        var count = checked((nint)BinaryPrimitives.ReadUInt32BigEndian(header.Slice(1 * sizeof(uint), sizeof(uint))));

        var labels = Tensor.CreateFromShapeUninitialized<byte>([(nint)count]);
        var labelsSpan = MemoryMarshal.CreateSpan(ref labels.GetPinnableReference(), checked((int)labels.FlattenedLength));
        gzStream.ReadExactly(labelsSpan);
        return labels;
    }

    public static Tensor<T> RepeatToCount<T>(Tensor<T> source, int count)
    {
        var sourceCount = checked((int)source.Lengths[0]);
        if (count <= sourceCount)
        {
            return source;
        }

        var sourceLengths = source.Lengths;
        var targetLengths = new nint[sourceLengths.Length];
        targetLengths[0] = count;
        for (var i = 1; i < sourceLengths.Length; i++)
        {
            targetLengths[i] = sourceLengths[i];
        }

        var target = Tensor.CreateFromShapeUninitialized<T>(targetLengths);
        var sourceSpan = MemoryMarshal.CreateSpan(ref source.GetPinnableReference(), checked((int)source.FlattenedLength));
        var targetSpan = MemoryMarshal.CreateSpan(ref target.GetPinnableReference(), checked((int)target.FlattenedLength));
        sourceSpan.CopyTo(targetSpan);

        var sampleSize = checked(sourceSpan.Length / sourceCount);
        for (var i = sourceCount; i < count;)
        {
            var samplesToCopy = Math.Min(sourceCount, count - i);
            var elementsToCopy = checked(samplesToCopy * sampleSize);
            sourceSpan[..elementsToCopy].CopyTo(targetSpan.Slice(checked(i * sampleSize), elementsToCopy));
            i += samplesToCopy;
        }

        return target;
    }
}
