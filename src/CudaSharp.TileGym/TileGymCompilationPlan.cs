using System.Collections.Generic;
using System.Linq;

namespace CudaSharp.Tester;

/// <summary>Identifies the source and signature shared by a kernel's template specializations.</summary>
sealed record TileGymKernelDefinition(string Header, string Name, string Signature);

/// <summary>Lists the concrete template specializations required for one kernel definition.</summary>
sealed class TileGymKernelRequest
{
    public TileGymKernelRequest(TileGymKernelDefinition definition, IEnumerable<string> templateArguments)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(templateArguments);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.Header);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.Name);
        ArgumentNullException.ThrowIfNull(definition.Signature);
        Definition = definition;
        var arguments = templateArguments.Distinct(StringComparer.Ordinal).ToArray();
        if (arguments.Length == 0)
        {
            throw new ArgumentException("A kernel request must contain specializations.", nameof(templateArguments));
        }
        var specializations = new TileGymKernelSpec[arguments.Length];
        for (var i = 0; i < arguments.Length; i++)
        {
            ArgumentNullException.ThrowIfNull(arguments[i], nameof(templateArguments));
            specializations[i] = new(definition.Header, definition.Name, arguments[i], definition.Signature);
        }
        Specializations = Array.AsReadOnly(specializations);
    }

    public TileGymKernelDefinition Definition { get; }
    public IReadOnlyList<TileGymKernelSpec> Specializations { get; }
}

/// <summary>A header compiled once with all requested kernels and their specializations.</summary>
sealed class TileGymCompilationUnit
{
    public TileGymCompilationUnit(string header, IEnumerable<TileGymKernelRequest> kernels)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(header);
        ArgumentNullException.ThrowIfNull(kernels);
        var requests = kernels.ToArray();
        if (requests.Length == 0 || requests.Any(kernel => kernel.Definition.Header != header))
        {
            throw new ArgumentException("A compilation unit must contain kernels from one header.", nameof(kernels));
        }
        Header = header;
        Kernels = Array.AsReadOnly(requests);
        var specializations = requests.SelectMany(static kernel => kernel.Specializations).Distinct().ToArray();
        Specializations = Array.AsReadOnly(specializations);
    }

    public string Header { get; }
    public IReadOnlyList<TileGymKernelRequest> Kernels { get; }
    public IReadOnlyList<TileGymKernelSpec> Specializations { get; }
}

/// <summary>Deduplicates required specializations into a header, kernel, specialization hierarchy.</summary>
sealed class TileGymCompilationPlan
{
    TileGymCompilationPlan(TileGymCompilationUnit[] units)
    {
        Units = Array.AsReadOnly(units);
    }

    public IReadOnlyList<TileGymCompilationUnit> Units { get; }

    public static TileGymCompilationPlan Create(IEnumerable<TileGymKernelRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var kernels = requests.GroupBy(static request => request.Definition)
            .Select(static group =>
            {
                var arguments = group.SelectMany(static request => request.Specializations)
                    .Select(static spec => spec.TemplateArguments);
                return new TileGymKernelRequest(group.Key, arguments);
            });
        var units = kernels.GroupBy(static request => request.Definition.Header, StringComparer.Ordinal)
            .Select(static group => new TileGymCompilationUnit(group.Key, group)).ToArray();
        return new(units);
    }

    public static TileGymCompilationPlan FromSpecs(IEnumerable<TileGymKernelSpec> specs)
    {
        ArgumentNullException.ThrowIfNull(specs);
        var requests = specs.GroupBy(static spec => new TileGymKernelDefinition(spec.Header, spec.Name, spec.Signature))
            .Select(static group =>
            {
                var arguments = group.Select(static spec => spec.TemplateArguments);
                return new TileGymKernelRequest(group.Key, arguments);
            });
        return Create(requests);
    }

    public TileGymCompilationPlan Where(Func<TileGymKernelSpec, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        var requests = new List<TileGymKernelRequest>();
        foreach (var unit in Units)
        {
            foreach (var kernel in unit.Kernels)
            {
                var arguments = kernel.Specializations.Where(predicate)
                    .Select(static spec => spec.TemplateArguments).ToArray();
                if (arguments.Length != 0)
                {
                    requests.Add(new(kernel.Definition, arguments));
                }
            }
        }
        return Create(requests);
    }
}
