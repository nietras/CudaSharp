using System;
using System.Collections.Generic;
using System.IO;
using CudaSharp.Tile;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;
using static CudaSharp.nvrtc;

namespace CudaSharp.Test;

[TestClass]
public class TileCppTest
{
    [TestMethod]
    public void TileCppTest_ConfigValidatesCompilerHints()
    {
        var config = CreateConfig(64, numCtas: 4, occupancy: 2);

        Assert.AreEqual("64", config["BLOCK_SIZE"]);
        Assert.AreEqual(4, config.NumCtas);
        Assert.AreEqual(2, config.Occupancy);
        Assert.AreEqual(3, config.OptimizationLevel);
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => CreateConfig(64, numCtas: 3));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => CreateConfig(64, occupancy: 33));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() =>
            new TileCppConfig([], numWorkerWarps: 5));
    }

    [TestMethod]
    public void TileCppTest_SearchSpaceRequiresMatchingParameterNames()
    {
        var first = CreateConfig(64);
        var second = new TileCppConfig([new("OTHER", "128")]);

        Assert.ThrowsExactly<ArgumentException>(() => new TileCppSearchSpace([first, second]));
    }

    [TestMethod]
    public void TileCppTest_AutotunerSelectsAndCachesFastestConfiguration()
    {
        var slow = CreateConfig(64);
        var fast = CreateConfig(128);
        var timer = new FakeTimer(new Dictionary<TileCppConfig, float>
        {
            [slow] = 2,
            [fast] = 1,
        });
        var tuner = new TileCppAutotuner(new TileCppSearchSpace([slow, fast]), timer);
        TileCppConfig? current = null;
        var launches = 0;

        void Launch(TileCppConfig config)
        {
            current = config;
            launches++;
        }

        timer.GetCurrent = () => current ?? throw new InvalidOperationException();
        var first = tuner.Tune(default, ("relu", 4096), Launch,
            static (_, config) => new TileCppGrid(4096 / uint.Parse(config["BLOCK_SIZE"])), seed: 42);
        var measuresAfterTuning = timer.MeasureCount;
        var second = tuner.Tune(default, ("relu", 4096), Launch,
            static (_, config) => new TileCppGrid(4096 / uint.Parse(config["BLOCK_SIZE"])), seed: 42);

        Assert.AreSame(fast, first.Config);
        Assert.AreEqual(new TileCppGrid(32), first.Grid);
        Assert.AreSame(first, second);
        Assert.AreEqual(2, measuresAfterTuning);
        Assert.AreEqual(measuresAfterTuning, timer.MeasureCount);
        Assert.AreEqual(1, timer.SynchronizeCount);
        Assert.IsGreaterThanOrEqualTo(6, launches);
    }

    [TestMethod]
    public void TileCppTest_CUDA13_3CompilesTileGymReluToTileIr()
    {
        try
        {
            nvrtcVersion(out var major, out var minor).Ok();
            if (major < 13 || major == 13 && minor < 3)
                Assert.Inconclusive($"CUDA Tile C++ requires NVRTC 13.3 or later; found {major}.{minor}.");

            var headerPath = Path.Combine(AppContext.BaseDirectory,
                "src-tilecpp", "tilegym", "activation", "relu.cuh");
            var header = new TileCppHeader("relu.cuh", File.ReadAllText(headerPath));
            const string source = """
                using int32_t = int;
                #include "relu.cuh"
                template __tile_global__ void relu_activation_fwd_kernel<float, 64, 0>(
                    const float*, float*, int, float, float, float, bool);
                """;
            var compiler = new TileCppCompiler(new RejectingAssembler(), nvrtcGetSupportedArchs()[^1]);

            var tileIr = compiler.CompileToTileIr(source, "relu.cu", new TileCppConfig([]), [header]);

            Assert.IsNotEmpty(tileIr);
        }
        catch (AssertInconclusiveException)
        {
            throw;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            Assert.Inconclusive($"CUDA Tile C++ NVRTC components are unavailable: {ex.Message}");
        }
    }

    static TileCppConfig CreateConfig(int blockSize, int? numCtas = null, int? occupancy = null) =>
        new([new("BLOCK_SIZE", blockSize.ToString())], numCtas, occupancy);

    sealed class RejectingAssembler : ITileIrAssembler
    {
        public byte[] Assemble(ReadOnlySpan<byte> tileIr, TileCppAssemblerOptions options) =>
            throw new AssertFailedException("TileIR assembly is not expected in this test.");
    }

    sealed class FakeTimer(IReadOnlyDictionary<TileCppConfig, float> timings) : ITileCppTimer
    {
        public Func<TileCppConfig>? GetCurrent { get; set; }
        public int MeasureCount { get; private set; }
        public int SynchronizeCount { get; private set; }

        public float Measure(Action launch, CUstream stream, TileCppTimingOptions options)
        {
            launch();
            MeasureCount++;
            return timings[GetCurrent!()];
        }

        public void Synchronize(CUstream stream) => SynchronizeCount++;
    }
}
