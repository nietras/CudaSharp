using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

static class TileGymTuningChecks
{
    public static void Run()
    {
        var problem = new TileGymMatmulProblem(129, 257, 65, "float", true, false, false, 120, 80);
        var candidate = TileGymMatmulCandidates.Create(64, 128, 32, occupancy: 2);
        Require(problem.Grid(candidate) == new TileCppGrid(9), "Normal matmul grid.");
        Require(problem.TemplateArguments(candidate) ==
            "float, 129, 257, 65, 64, 128, 32, 8, 3, true, false, 1, 2", "Normal template order.");
        var persistent = problem with { Persistent = true };
        Require(persistent.Grid(candidate) == new TileCppGrid(18), "Persistent matmul grid.");
        Require(persistent.TemplateArguments(candidate) ==
            "float, 129, 257, 65, 64, 128, 32, 8, true, false, 1, 2", "Persistent template order.");
        Require(TileGymMatmulCandidates.For(problem).Count == 2, "SM120 standard candidates.");
        Require(TileGymMatmulCandidates.For(persistent).Count == 7, "SM120 persistent candidates.");
        Require(TileGymMatmulCandidates.For(problem with { Architecture = 80 }).Count == 24,
            "Pre-SM90 standard candidates.");
        Require(TileGymMatmulCandidates.For(persistent with { Architecture = 80 }).Count == 5,
            "Pre-SM90 persistent candidates.");
        Require(TileGymMatmulCandidates.For(problem with { Architecture = 100 }).Count == 4,
            "Blackwell standard candidates.");
        Require(TileGymMatmulCandidates.For(persistent with { Architecture = 100 }).Count == 5,
            "Blackwell persistent candidates.");
        Require(TileGymMatmulCandidates.Select(problem, null, 32, null, 3).Count == 1,
            "Explicit overrides select one candidate.");
        Require(TileGymMatmulCandidates.Select(problem, null, 32, null, 3)[0]["Occupancy"] == "3",
            "Override value.");
        VerifyFamilies();
        VerifyConvolutions();
        VerifyTuner();
    }

    static void VerifyFamilies()
    {
        var bmm = new TileGymBmmProblem(2, 64, 64, 64, "float", false, false, true, 120, 80);
        var bmmCandidates = TileGymBmmCandidates.For(bmm);
        Require(bmmCandidates.Count == 24, "SM120 persistent BMM candidates.");
        Require(bmm.Grid(bmmCandidates[0]) == new TileCppGrid(2), "Persistent BMM grid.");
        Require(bmm.TemplateArguments(bmmCandidates[0]) ==
            "float, 64, 64, 32, 8, false, false, 2, 64, 64, 64, 1, 1", "Persistent BMM template order.");
        Require((bmm with { Persistent = false }).Grid(TileGymBmmCandidates.For(bmm with { Persistent = false })[0]) ==
            new TileCppGrid(1, 2), "Normal BMM grid.");

        var forward = new TileGymAttentionProblem(TileGymAttentionKind.Forward, 64, 64, true, 120);
        Require(TileGymAttentionCandidates.For(forward).Count == 2, "SM120 forward attention candidates.");
        Require(forward.TemplateArguments(TileGymAttentionCandidates.For(forward)[0]) ==
            "float, 1, 1, 1, 64, 64, 64, 64, 64, true, true, 2, 1", "Forward attention template order.");
        var gemma = forward with { Kind = TileGymAttentionKind.GemmaForward };
        Require(TileGymAttentionCandidates.For(gemma).Count == 2, "Gemma fallback candidates.");
        Require(gemma.TemplateArguments(TileGymAttentionCandidates.For(gemma)[0]) ==
            "float, 1, 1, 1, 64, 64, 64, 64, 64, true, 0, false, 2", "Gemma template order.");
        var backward = forward with { Kind = TileGymAttentionKind.BackwardMain };
        Require(backward.Grid(TileGymAttentionCandidates.For(backward)[0]) == new TileCppGrid(1),
            "Attention backward grid.");
        try
        {
            TileGymAttentionCandidates.For(forward with { Sequence = 32 });
            throw new InvalidOperationException("Undersized attention sequence was accepted.");
        }
        catch (NotSupportedException)
        {
        }
        try
        {
            TileGymAttentionCandidates.For(backward with { Sequence = 96 });
            throw new InvalidOperationException("Unaligned backward attention sequence was accepted.");
        }
        catch (NotSupportedException)
        {
        }

        var layerNorm = new TileGymPersistentLayerNormProblem(8, 256, 80);
        Require(TileGymPersistentLayerNormCandidates.For(layerNorm).Count == 3,
            "Persistent layer norm row pruning.");
        Require(layerNorm.TemplateArguments(TileGymPersistentLayerNormCandidates.For(layerNorm)[0]) ==
            "float, float, float, 2, 256, false, true, true, 8, 256, 4, 0.00001f",
            "Persistent layer norm grid in template.");
        var rmsNorm = new TileGymPersistentRmsNormProblem(8, 256, 80);
        Require(TileGymPersistentRmsNormCandidates.For(rmsNorm).Count == 6, "Persistent RMS norm variants.");
        Require(rmsNorm.Grid(TileGymPersistentRmsNormCandidates.For(rmsNorm)[0]) == new TileCppGrid(4),
            "Persistent RMS norm grid.");

        var elementwise = new TileGymElementwiseProblem(1025, 0);
        Require(elementwise.Grid(TileGymElementwiseCandidates.For(1025)[0]) == new TileCppGrid(5),
            "Elementwise masked tail grid.");
        foreach (var operation in Enum.GetValues<TileGymReluOperation>())
        {
            var specialization = elementwise with { Operation = (int)operation };
            Require(specialization.TemplateArguments(TileGymElementwiseCandidates.For(1025)[0]) ==
                $"float, 256, {(int)operation}", "Activation OP template argument.");
            Require(float.IsFinite(TileGymActivationScenarios.ReluReference(-1f, operation, .5f, .125f,
                1f / 3, false)), "Activation forward reference.");
            Require(float.IsFinite(TileGymActivationScenarios.ReluReference(-1f, operation, .5f, .125f,
                1f / 3, true)), "Activation backward reference.");
        }
        var softmax = new TileGymSoftmaxProblem(4, 256, false, false);
        Require(TileGymSoftmaxCandidates.For(softmax).Count == 2, "Single-pass softmax pruning.");
        Require(softmax.TemplateArguments(TileGymSoftmaxCandidates.For(softmax)[0]) == "float, 256, 0",
            "Single-pass softmax template arguments.");
        Require((softmax with { Online = true }).TemplateArguments(TileGymSoftmaxCandidates.For(softmax with { Online = true })[0]) == "float, 128",
            "Online softmax template arguments.");
        Require(TileGymSoftmaxCandidates.For(softmax with { Online = true }).Count == 3,
            "Online softmax variants.");
        var dropout = new TileGymDropoutProblem(4097, .25f, 2654435761u);
        Require(dropout.Grid(TileGymDropoutCandidates.For()[0]) == new TileCppGrid(17),
            "Dropout masked tail grid.");
        Require(dropout.TemplateArguments(TileGymDropoutCandidates.For()[0]) ==
            "float, 256, 4097, 0.25f, 2654435761u", "Dropout template arguments.");
    }

    static void VerifyConvolutions()
    {
        var problems = TileGymConvolutionScenarios.Problems;
        Require(problems.Length == 4, "Four convolution examples.");
        foreach (var problem in problems)
        {
            Require(problem.OutputLength > 0 && problem.Grid(128).X ==
                (problem.OutputLength + 127) / 128, "Convolution output grid.");
            Require(problem.TemplateArguments(128).EndsWith($"{problem.Groups}, 128", StringComparison.Ordinal),
                "Convolution template dimensions.");
            Require(problem.MmaGrid().X == (problem.N * problem.Od * problem.Oh * problem.Ow + 31) / 32 &&
                problem.MmaGrid().Y == problem.Groups * ((problem.Co / problem.Groups + problem.MmaTileN - 1) / problem.MmaTileN),
                "Grouped convolution MMA grid.");
            var input = new float[problem.InputLength];
            var weights = new float[problem.WeightLength];
            Array.Fill(input, 1f);
            Array.Fill(weights, 1f);
            var bias = new float[problem.Co];
            var modelBias = new float[problem.Co];
            var result = problem.Reference(input, weights, bias, modelBias);
            Require(result.Length == problem.OutputLength && Array.TrueForAll(result, float.IsFinite),
                "Convolution CPU reference.");
        }
        var forward = problems[0];
        var x = new float[forward.InputLength];
        var w = new float[forward.WeightLength];
        x[0] = 2;
        w[0] = 3;
        var cb = new float[forward.Co];
        var mb = new float[forward.Co];
        cb[0] = -1;
        mb[0] = 2;
        Require(Array.Exists(forward.Reference(x, w, cb, mb), v => v == 2),
            "Forward convolution bias, ReLU and model bias order.");
        Require(problems[2].OutputPaddingH == 1 && problems[3].OutputPaddingD == 1,
            "Transposed convolution output padding.");
        var boundary = TileGymConvolutionScenarios.MmaBoundaryProblem;
        Require(boundary.WeightLength / boundary.Co > 64 && boundary.Co / boundary.Groups > 64 &&
            boundary.MmaGrid().Y == 4, "Grouped MMA K and N tail coverage.");
        Require(TileGymConvolutionScenarios.ToBfloat16(1f) == 0x3f80 &&
            TileGymConvolutionScenarios.FromBfloat16(0x3f80) == 1f &&
            TileGymConvolutionScenarios.ToBfloat16(BitConverter.UInt32BitsToSingle(0x3f808000)) == 0x3f80,
            "BF16 packing and ties-to-even rounding.");
        Require(TileGymConvolutionScenarios.ToBfloat16(BitConverter.UInt32BitsToSingle(0x3f818000)) == 0x3f82 &&
            TileGymConvolutionScenarios.ToBfloat16(float.PositiveInfinity) == 0x7f80 &&
            TileGymConvolutionScenarios.ToBfloat16(float.NegativeInfinity) == 0xff80,
            "BF16 odd ties and infinities.");
        Require(float.IsNaN(TileGymConvolutionScenarios.FromBfloat16(
                TileGymConvolutionScenarios.ToBfloat16(BitConverter.UInt32BitsToSingle(0x7f800001)))) &&
            float.IsNaN(TileGymConvolutionScenarios.FromBfloat16(
                TileGymConvolutionScenarios.ToBfloat16(BitConverter.UInt32BitsToSingle(0xff800001)))),
            "BF16 positive and negative NaNs retain a payload.");
        TileGymConvolutionScenarios.ValidateMmaOutput([0x3dcd], [0.1f], true, "bf16 one ULP");
        try
        {
            TileGymConvolutionScenarios.ValidateMmaOutput([0x3dd0], [0.1f], true, "bf16 drift");
            throw new InvalidOperationException("BF16 validation accepted an output beyond one ULP.");
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("more than one output ULP", StringComparison.Ordinal))
        {
        }
        TileGymConvolutionScenarios.ValidateMmaOutput([0x2e67], [0.1f], false, "fp16 one ULP");
        try
        {
            TileGymConvolutionScenarios.ValidateMmaOutput([0x2e70], [0.1f], false, "fp16 drift");
            throw new InvalidOperationException("FP16 validation accepted an output beyond one ULP.");
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("more than one output ULP", StringComparison.Ordinal))
        {
        }
    }

    static void VerifyTuner()
    {
        var problem = new TileGymMatmulProblem(128, 128, 64, "float", false, false, true, 120, 2);
        var rejected = TileGymMatmulCandidates.Create(64, 64, 32);
        var slow = TileGymMatmulCandidates.Create(64, 64, 64);
        var fast = TileGymMatmulCandidates.Create(128, 64, 64);
        var invalid = TileGymMatmulCandidates.Create(128, 128, 64, numCtas: 4);
        var kernel = TileGymKernel.Tuned("matmul.cuh", "check_kernel", "const float*, const float*, float*",
            problem, [rejected, slow, invalid, fast]);
        Require(kernel.Variants.Count == 3, "Candidates invalid for the problem are dropped when declared.");
        var rejectedSpec = kernel.Variants[0].Spec;
        var fastSpec = kernel.Variants[2].Spec;
        var fastFunction = new CUfunction(2);
        TileGymCompiledKernel Resolve(TileGymKernelSpec spec) => spec == rejectedSpec
            ? new(spec, default, 1, 2, new InvalidOperationException("Rejected specialization."))
            : new(spec, spec == fastSpec ? fastFunction : new CUfunction(1), 1, 2);
        var timer = new FakeTimer();
        var tuner = new TileGymTuner(timer, default, Resolve);
        var launches = 0;
        var validations = 0;
        void Launch(CUfunction function, TileCppGrid grid)
        {
            Require(grid.X > 0, "Candidate grid.");
            launches++;
            timer.Fast = function == fastFunction;
        }
        var selection = tuner.Select(kernel.Name, kernel.Variants, Launch, () => validations++);
        Require(selection.Variant.Spec == fastSpec, "Fastest candidate selected.");
        Require(selection.Candidates == 3 && selection.Rejections.Count == 1, "Rejected candidates are tracked.");
        Require(validations == 2 && timer.MeasureCount == 2 && launches == 4,
            "Each loaded candidate is validated once and timed once.");
        Require(selection.FirstLaunchMilliseconds >= 0 && selection.TuneMilliseconds >= 0, "Tuning times.");

        var single = TileGymKernel.Fixed("matmul.cuh", "check_kernel", "fixed", "const float*", new TileCppGrid(1));
        var measured = timer.MeasureCount;
        var only = tuner.Select(single.Name, single.Variants, Launch);
        Require(only.TuneMilliseconds == 0 && timer.MeasureCount == measured && only.Candidates == 1,
            "A single variant is not timed during tuning.");

        var report = new TileGymReport();
        report.Add("matmul", "128x128", 1024, new TileGymRun(kernel, selection, 1, .5));
        Require(report.Results[0].Status == "Passed (searched)" &&
            report.Results[0].Diagnostic?.StartsWith("3 candidates, 1 rejected.", StringComparison.Ordinal) == true,
            "Search report does not claim every candidate passed.");
        Require(report.Results[0].CompileMilliseconds == 1 && report.Results[0].LoadMilliseconds == 2 &&
            report.Results[0].FirstLaunchMilliseconds == selection.FirstLaunchMilliseconds &&
            report.Results[0].HostMilliseconds == .5 && report.Results[0].KernelMilliseconds == 1,
            "Phase timings are included in reports.");
        report.Add("activation", "1024", 1024, new TileGymRun(single, only, 6, 5));
        Require(report.Results[1].Status == "Passed" && report.Results[1].Diagnostic is null,
            "Fixed kernels report plain status.");
        Require(report.ToCsv().Contains("\"1\",\"2\",", StringComparison.Ordinal) &&
            report.ToMarkdown().Contains("| Load ms | First launch ms | Tune ms | Host ms | Kernel ms |",
                StringComparison.Ordinal),
            "Phase timings appear in CSV and Markdown reports.");
    }

    static void Require(bool condition, string description)
    {
        if (!condition)
        {
            throw new InvalidOperationException($"TileGym tuning check failed: {description}");
        }
    }

    sealed class FakeTimer : ITileCppTimer
    {
        public bool Fast { get; set; }
        public int MeasureCount { get; private set; }
        public void Synchronize(CUstream stream) { }
        public float Measure(Action launch, CUstream stream, TileCppTimingOptions options)
        {
            launch();
            MeasureCount++;
            return Fast ? 1 : 2;
        }
    }
}
