using System;

namespace CudaSharp.Mnist;

public readonly record struct ParamGroup(
    string Name, int OutFeatures, int InFeatures,
    int WeightOffset, int BiasOffset)
{
    public int WeightCount => OutFeatures * InFeatures;
    public int BiasCount => OutFeatures;
    public int TotalCount => WeightCount + BiasCount;
}

public sealed class NetworkConfig
{
    public required string Name { get; init; }
    public required string CudaSource { get; init; }

    // Conv1 layer
    public required int Conv1FilterCount { get; init; }
    public required int Conv1FilterSize { get; init; }

    // Conv2 layer
    public required int Conv2FilterCount { get; init; }
    public required int Conv2FilterSize { get; init; }
    public required int Pool2OutSize { get; init; }

    // FC layers
    public required bool HasFC1 { get; init; }
    public int FC1Outputs { get; init; } = 256;

    // Training schedule
    public required int BatchesPerEpoch { get; init; }
    public required int TotalSteps { get; init; }

    // Hyperparameters
    public required float MaxLR { get; init; }
    public float Beta1 { get; init; } = 0.7f;
    public float Beta2 { get; init; } = 0.9f;

    // Computed architecture sizes
    public int Conv1OutPerSample => 12 * 12 * Conv1FilterCount;
    public int Conv1UnpooledPerSample => 24 * 24 * Conv1FilterCount;
    public int Conv2OutPerSample => Pool2OutSize * Pool2OutSize * Conv2FilterCount;

    public int Conv2UnpooledPerSample
    {
        get
        {
            int s = Pool2OutSize * 2;
            return s * s * Conv2FilterCount;
        }
    }

    public int FC1Inputs => Conv2OutPerSample;
    public int FC2Inputs => HasFC1 ? FC1Outputs : Conv2OutPerSample;

    // Contiguous parameter layout
    ParamGroup[]? _paramGroups;

    public ParamGroup[] ParamGroups => _paramGroups ??= BuildParamLayout();

    public int TotalParamElements
    {
        get
        {
            int total = 0;
            foreach (var pg in ParamGroups)
            {
                total += pg.TotalCount;
            }
            return total;
        }
    }

    public ParamGroup GetParam(string name)
    {
        foreach (var pg in ParamGroups)
        {
            if (pg.Name == name) return pg;
        }
        throw new ArgumentException($"Unknown param group: {name}", nameof(name));
    }

    ParamGroup[] BuildParamLayout()
    {
        int count = HasFC1 ? 4 : 3;
        var result = new ParamGroup[count];
        int offset = 0;
        int idx = 0;

        void Add(string name, int outF, int inF)
        {
            int wc = outF * inF;
            result[idx++] = new ParamGroup(name, outF, inF, offset, offset + wc);
            offset += wc + outF;
        }

        Add("conv1", Conv1FilterCount, 1 * Conv1FilterSize * Conv1FilterSize);
        Add("conv2", Conv2FilterCount, Conv1FilterCount * Conv2FilterSize * Conv2FilterSize);

        if (HasFC1)
        {
            Add("fc1", FC1Outputs, FC1Inputs);
        }

        Add("fc2", 10, FC2Inputs);
        return result;
    }
}
