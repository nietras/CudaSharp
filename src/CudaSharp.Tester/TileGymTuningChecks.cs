using System;
using CudaSharp.Tile;
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
        VerifySession();
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
        Require(TileGymSoftmaxCandidates.For(softmax with { Online = true }).Count == 3,
            "Online softmax variants.");
        Require(new TileGymDropoutProblem(4097).Grid(TileGymDropoutCandidates.For()[0]) == new TileCppGrid(17),
            "Dropout masked tail grid.");
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
    }

    static void VerifySession()
    {
        var problem = new TileGymMatmulProblem(128, 128, 64, "float", false, false, false, 120, 80);
        var rejected = TileGymMatmulCandidates.Create(64, 64, 32);
        var slow = TileGymMatmulCandidates.Create(64, 64, 64);
        var fast = TileGymMatmulCandidates.Create(128, 64, 64);
        var timer = new FakeTimer();
        var launches = 0;
        var creations = 0;
        var validations = 0;
        using var session = new TileGymTuningSession<TileGymMatmulProblem>([rejected, slow, fast],
            (_, candidate) =>
            {
                if (ReferenceEquals(candidate, rejected))
                {
                    throw new InvalidOperationException("Rejected specialization.");
                }
                creations++;
                return new TileCppKernel(new TileCppCompiler(120, installBundledHeaders: false),
                    "", "check.cu", "check_kernel");
            },
            static (p, candidate) => p.Grid(candidate), timer);
        void Launch(TileCppKernel _, TileCppConfig config, TileCppGrid grid)
        {
            Require(grid.X > 0, "Candidate grid.");
            launches++;
            timer.LastConfig = config;
        }
        timer.FastConfig = fast.CompilerConfig;
        var first = session.Tune(problem, default, Launch,
            validate: _ => validations++);
        var measurements = timer.MeasureCount;
        var second = session.Tune(problem, default, Launch,
            validate: _ => validations++);
        Require(ReferenceEquals(first.Candidate, fast), "Fastest candidate selected.");
        Require(second.CacheHit && !first.CacheHit, "Repeated tuning reuses winner.");
        Require(second.TuneMilliseconds == 0 && timer.MeasureCount == measurements,
            "Cached tuning does not benchmark again.");
        Require(creations == 2 && launches > measurements, "Only successful kernels are created and launched.");
        Require(validations == 2, "Each valid candidate is checked once, including cached winner.");
        Require(first.CandidateCount == 3, "Report tracks candidate count.");
        var report = new TileGymReport();
        TileGymKernel.Report(report, "matmul", "check_kernel", "128x128", 1024,
            (0d, 1d), first);
        Require(report.Results[0].Status == "Passed (searched)" &&
            report.Results[0].Diagnostic?.Contains("3 candidates offered", StringComparison.Ordinal) == true,
            "Search report does not claim every candidate passed.");
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
        public TileCppConfig? FastConfig { get; set; }
        public TileCppConfig? LastConfig { get; set; }
        public int MeasureCount { get; private set; }
        public void Synchronize(CUstream stream) { }
        public float Measure(Action launch, CUstream stream, TileCppTimingOptions options)
        {
            launch();
            MeasureCount++;
            return ReferenceEquals(LastConfig, FastConfig) ? 1 : 2;
        }
    }
}
