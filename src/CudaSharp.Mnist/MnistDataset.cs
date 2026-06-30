using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
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


    public static (uint[] images, int count) ParseImagesGz(string filePath, int maxCount)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gzStream.CopyTo(ms);
        var bytes = ms.ToArray();

        var magic = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(0 * sizeof(uint), sizeof(uint)));
        if (magic != 0x00000803)
            throw new InvalidOperationException($"Invalid images magic number: {magic:X}");

        var count = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(1 * sizeof(uint), sizeof(uint)));
        var rows = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(2 * sizeof(uint), sizeof(uint)));
        var cols = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(3 * sizeof(uint), sizeof(uint)));

        if (rows != 28 || cols != 28)
            throw new InvalidOperationException($"Expected 28x28 images, but got {rows}x{cols}");

        var imageCountToLoad = maxCount;
        var packedImages = new uint[imageCountToLoad * 28];

        for (var i = 0; i < imageCountToLoad; i++)
        {
            var sourceImageIdx = i % count;
            var sourcePixelOffset = 16 + sourceImageIdx * 28 * 28;

            for (var r = 0; r < 28; r++)
            {
                uint rowBits = 0;
                for (var c = 0; c < 28; c++)
                {
                    var pixelVal = bytes[sourcePixelOffset++];
                    if (pixelVal > 127)
                    {
                        rowBits |= (1u << c);
                    }
                }
                packedImages[i * 28 + r] = rowBits;
            }
        }

        return (packedImages, imageCountToLoad);
    }

    public static int[] ParseLabelsGz(string filePath, int maxCount)
    {
        using var fileStream = File.OpenRead(filePath);
        using var gzStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gzStream.CopyTo(ms);
        var bytes = ms.ToArray();

        var magic = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(0 * sizeof(uint), sizeof(uint)));
        if (magic != 0x00000801)
            throw new InvalidOperationException($"Invalid images magic number: {magic:X}");

        var count = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(1 * sizeof(uint), sizeof(uint)));

        var labelCountToLoad = maxCount;
        var labels = new int[labelCountToLoad];

        for (var i = 0; i < labelCountToLoad; i++)
        {
            labels[i] = bytes[8 + (i % count)];
        }

        return labels;
    }
}
