using System;
using System.Collections.Generic;

namespace CudaSharp.Mnist;

public unsafe partial class Program
{
    public static readonly NetworkConfig ConfigV1 = new()
    {
        Name = "V1",
        CudaSource = CudaSourceV1,
        IsHalf = false,
        IsV7Based = false,
        Conv1FilterCount = 16,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 5,
        HasFC1 = true,
        FC1Outputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 400,
        MaxLR = 0.003f
    };

    public static readonly NetworkConfig ConfigV2 = new()
    {
        Name = "V2",
        CudaSource = CudaSourceV2,
        IsHalf = false,
        IsV7Based = false,
        Conv1FilterCount = 8,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = false,
        BatchesPerEpoch = 300,
        TotalSteps = 600,
        MaxLR = 0.06f
    };

    public static readonly NetworkConfig ConfigV3 = new()
    {
        Name = "V3",
        CudaSource = CudaSourceV3,
        IsHalf = true,
        IsV7Based = false,
        Conv1FilterCount = 8,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = false,
        BatchSize = 128,
        BatchesPerEpoch = 200,
        TotalSteps = 200,
        MaxLR = 0.05f
    };

    public static readonly NetworkConfig ConfigV4 = new()
    {
        Name = "V4",
        CudaSource = CudaSourceV4,
        IsHalf = false,
        IsV7Based = false,
        BatchSize = 128,
        Conv1FilterCount = 16,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 5,
        HasFC1 = true,
        FC1Outputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 190,
        MaxLR = 0.003f
    };

    public static readonly NetworkConfig ConfigV5 = new()
    {
        Name = "V5",
        CudaSource = CudaSourceV5,
        IsHalf = false,
        IsV7Based = false,
        BatchSize = 128,
        Conv1FilterCount = 1,
        Conv1FilterSize = 1,
        Conv2FilterCount = 1,
        Conv2FilterSize = 1,
        Pool2OutSize = 1,
        HasFC1 = true,
        FC1Outputs = 256,
        BatchesPerEpoch = 180,
        TotalSteps = 180,
        MaxLR = 0.009f
    };

    public static readonly NetworkConfig ConfigV6 = new()
    {
        Name = "V6",
        CudaSource = CudaSourceV6,
        IsHalf = true,
        IsV7Based = false,
        BatchSize = 128,
        Conv1FilterCount = 16,
        Conv1FilterSize = 5,
        Conv2FilterCount = 32,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = false,
        BatchesPerEpoch = 240,
        TotalSteps = 240,
        MaxLR = 0.007f
    };

    public static readonly NetworkConfig ConfigV7 = new()
    {
        Name = "V7",
        CudaSource = CudaSourceV7,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 8,
        Conv1FilterSize = 3,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 5,
        HasFC1 = false,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV8 = new()
    {
        Name = "V8",
        CudaSource = CudaSourceV8,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 16,
        Conv1FilterSize = 3,
        Conv2FilterCount = 16,
        Conv2FilterSize = 3,
        Pool2OutSize = 7,
        HasFC1 = true,
        FC1Outputs = 16,
        FC1Inputs = 144,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV9 = new()
    {
        Name = "V9",
        CudaSource = CudaSourceV9,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV10 = new()
    {
        Name = "V10",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 256,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 200,
        TotalSteps = 300,
        MaxLR = 0.006f
    };

    public static readonly NetworkConfig ConfigV11 = new()
    {
        Name = "V11",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 400,
        TotalSteps = 155,
        MaxLR = 0.014f
    };

    public static readonly NetworkConfig ConfigV12 = new()
    {
        Name = "V12",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 400,
        TotalSteps = 155,
        MaxLR = 0.016f
    };

    public static readonly NetworkConfig ConfigV13 = new()
    {
        Name = "V13",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 400,
        TotalSteps = 100,
        MaxLR = 0.010f
    };

    public static readonly NetworkConfig ConfigV14 = new()
    {
        Name = "V14",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 400,
        TotalSteps = 80,
        MaxLR = 0.025f
    };

    public static readonly NetworkConfig ConfigV20 = new()
    {
        Name = "V20",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 6,
        Conv1FilterSize = 5,
        Conv2FilterCount = 16,
        Conv2FilterSize = 5,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 400,
        TotalSteps = 200,
        MaxLR = 0.005f,
        ActivationType = "SILU"
    };

    public static readonly NetworkConfig ConfigV21 = new()
    {
        Name = "V21",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 8,
        Conv1FilterSize = 2,
        Conv2FilterCount = 16,
        Conv2FilterSize = 2,
        Pool2OutSize = 6,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 576,
        BatchesPerEpoch = 400,
        TotalSteps = 155,
        MaxLR = 0.014f
    };

    public static readonly NetworkConfig ConfigV22 = new()
    {
        Name = "V22",
        CudaSource = CudaSourceV10,
        IsHalf = true,
        IsV7Based = true,
        BatchSize = 128,
        Conv1FilterCount = 8,
        Conv1FilterSize = 4,
        Conv2FilterCount = 16,
        Conv2FilterSize = 4,
        Pool2OutSize = 4,
        HasFC1 = true,
        FC1Outputs = 120,
        FC1Inputs = 256,
        BatchesPerEpoch = 400,
        TotalSteps = 155,
        MaxLR = 0.014f
    };

    public static readonly IReadOnlyList<NetworkConfig> OrderedNetworkConfigs =
    [
        ConfigV1,
        ConfigV2,
        ConfigV3,
        ConfigV4,
        ConfigV5,
        ConfigV6,
        ConfigV7,
        ConfigV8,
        ConfigV9,
        ConfigV10,
        ConfigV11,
        ConfigV12,
        ConfigV13,
        ConfigV14,
        ConfigV20,
        ConfigV21,
        ConfigV22
    ];

    public static readonly IReadOnlyDictionary<string, NetworkConfig> NetworkConfigs = CreateNetworkConfigs();

    static IReadOnlyDictionary<string, NetworkConfig> CreateNetworkConfigs()
    {
        var configs = new Dictionary<string, NetworkConfig>(OrderedNetworkConfigs.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var config in OrderedNetworkConfigs)
        {
            configs.Add(config.Name, config);
        }
        return configs;
    }
}
