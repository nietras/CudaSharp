using System.Collections.Generic;
using CudaSharp.TileGym;
using static CudaSharp.nvcuda;

namespace CudaSharp.Tester;

/// <summary>Enqueues one launch of a loaded kernel variant on the runtime stream.</summary>
delegate void TileGymLaunch(CUfunction function, TileCppGrid grid);

/// <summary>Maps tuning candidates to template arguments and launch grids for one problem.</summary>
interface ITileGymProblem
{
    string TemplateArguments(TileGymCandidate candidate);
    TileCppGrid Grid(TileGymCandidate candidate);
}

/// <summary>Identifies one compiled CUDA Tile C++ kernel specialization.</summary>
sealed record TileGymKernelSpec(string Header, string Name, string TemplateArguments, string Signature)
{
    public override string ToString() => $"{Name}<{TemplateArguments}>";
}

/// <summary>One launchable specialization of a kernel together with its grid and report label.</summary>
sealed record TileGymVariant(TileGymKernelSpec Spec, TileCppGrid Grid, string Label);

/// <summary>
/// Declares a kernel as data, so all specializations can be compiled and loaded before any launch. Fixed kernels
/// have a single variant; tuned kernels have one variant per valid candidate.
/// </summary>
sealed class TileGymKernel
{
    TileGymKernel(string name, IReadOnlyList<TileGymVariant> variants)
    {
        Name = name;
        Variants = variants;
    }

    public string Name { get; }
    public IReadOnlyList<TileGymVariant> Variants { get; }

    public static TileGymKernel Fixed(string header, string name, string templateArguments, string signature,
        TileCppGrid grid, string? label = null)
    {
        var spec = new TileGymKernelSpec(header, name, templateArguments, signature);
        return new(name, [new TileGymVariant(spec, grid, label ?? templateArguments)]);
    }

    public static TileGymKernel Tuned<TProblem>(string header, string name, string signature,
        TProblem problem, IEnumerable<TileGymCandidate> candidates)
        where TProblem : ITileGymProblem
    {
        ArgumentNullException.ThrowIfNull(candidates);
        var variants = new List<TileGymVariant>();
        foreach (var candidate in candidates)
        {
            try
            {
                var spec = new TileGymKernelSpec(header, name, problem.TemplateArguments(candidate), signature);
                variants.Add(new TileGymVariant(spec, problem.Grid(candidate), candidate.ToString()));
            }
            catch (Exception ex) when (ex is ArgumentException or OverflowException)
            {
                // The candidate is not valid for this problem shape.
            }
        }
        if (variants.Count == 0)
        {
            throw new ArgumentException($"No candidate is valid for {name}.", nameof(candidates));
        }
        return new(name, variants);
    }
}

/// <summary>Declares the kernels a benchmark needs and how to run it once they are compiled and loaded.</summary>
sealed record TileGymBenchmark(IReadOnlyList<TileGymKernel> Kernels, Action<TileGymRuntime, TileGymReport> Run)
{
    public TileGymBenchmark(TileGymKernel kernel, Action<TileGymRuntime, TileGymReport> run)
        : this([kernel], run) { }
}
