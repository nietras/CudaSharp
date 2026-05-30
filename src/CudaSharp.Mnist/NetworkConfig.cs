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
    public bool IsHalf { get; init; } = true;
    public bool IsV7Based { get; init; } = false;
    public bool UseCustomCudaSource { get; init; } = false;
    public string FC1ForwardKernelName { get; init; } = "fc1_forward";
    public string FC1BackwardKernelName { get; init; } = "fc1_backward";
    public string FC1BackwardWeightsKernelName { get; init; } = "fc1_backward_weights";
    public bool UsesPooledConv2AsFc1Input { get; init; } = false;
    public int? FC1OutputElementsOverride { get; init; }
    public int? FC2InputsOverride { get; init; }
    public bool RequiresIntermediateGradBuffer { get; init; } = false;
    public bool UseCustomEvaluationPath { get; init; } = false;

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
    public int BatchSize { get; init; } = 128;
    public required int BatchesPerEpoch { get; init; }
    public required int TotalSteps { get; init; }

    // Hyperparameters
    public required float MaxLR { get; init; }
    public float Beta1 { get; init; } = 0.7f;
    public float Beta2 { get; init; } = 0.9f;

    // Capability flags (replace version-name string checks)
    public bool HasFc1Unpooled { get; init; } = false;
    public bool HasSeparateBackwardWeights { get; init; } = false;
    public float FC1WeightScale { get; init; } = 1.0f;
    public bool IsResNet { get; init; } = false;
    public bool IsMlpOnly { get; init; } = false;

    // Modern Techniques
    public bool HasDropout { get; init; } = false;
    public float DropoutRate { get; init; } = 0.5f;
    public bool HasWeightDecay { get; init; } = false;
    public float WeightDecayRate { get; init; } = 0.01f;
    public bool IsFusedForward { get; init; } = false;
    public bool HasLayerNorm { get; init; } = false;
    public bool IsGlobalAveragePooling { get; init; } = false;
    public string ActivationType { get; init; } = "GELU";

    // Computed architecture sizes
    public int Conv1OutSize => 28 - Conv1FilterSize + 1;
    public int Pool1OutSize => Conv1OutSize / 2;
    public int Conv1OutPerSample => Pool1OutSize * Pool1OutSize * Conv1FilterCount;
    public int Conv1UnpooledPerSample => UsesPooledConv2AsFc1Input ? 3136 : Conv1OutSize * Conv1OutSize * Conv1FilterCount;
    public int Conv2OutSize => Pool1OutSize - Conv2FilterSize + 1;
    public int Conv2OutPerSample => UsesPooledConv2AsFc1Input ? 3136 : Pool2OutSize * Pool2OutSize * Conv2FilterCount;

    public int Conv2UnpooledPerSample
    {
        get
        {
            int s = Pool2OutSize * 2;
            return s * s * Conv2FilterCount;
        }
    }

    int _fc1Inputs;
    public int FC1Inputs
    {
        get => _fc1Inputs > 0 ? _fc1Inputs : ((Name == "V5" || Name == "V6") ? 784 : Conv2OutPerSample);
        init => _fc1Inputs = value;
    }
    public int FC2Inputs => FC2InputsOverride ?? (HasFC1 ? FC1Outputs : Conv2OutPerSample);

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
            if (HasLayerNorm)
            {
                // LayerNorm has gamma (weight) and beta (bias). Both are size FC1Outputs.
                // We'll map Gamma to OutFeatures (BiasCount) and Beta to InFeatures (WeightCount) or vice-versa?
                // Actually, ParamGroup is OutFeatures * InFeatures + OutFeatures.
                // If OutFeatures = FC1Outputs, InFeatures = 1, then WeightCount = FC1Outputs, BiasCount = FC1Outputs. Perfect!
                Add("fc1_ln", FC1Outputs, 1);
            }
        }

        Add("fc2", 10, FC2Inputs);
        return result;
    }
}
