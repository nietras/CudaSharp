using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;

namespace CudaSharp.Tester;

readonly record struct TileGymHyperparameter(string Name, string CppValue)
{
    public static TileGymHyperparameter Integer(string name, int value)
        => new(name, value.ToString(CultureInfo.InvariantCulture));

    public static TileGymHyperparameter Boolean(string name, bool value)
        => new(name, value ? "true" : "false");
}

sealed class TileGymCandidate
{
    readonly IReadOnlyList<TileGymHyperparameter> _parameters;
    readonly IReadOnlyDictionary<string, string> _values;

    public TileGymCandidate(IEnumerable<TileGymHyperparameter> parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        var items = parameters.ToArray();
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var item in items)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(item.Name);
            ArgumentException.ThrowIfNullOrWhiteSpace(item.CppValue);
            if (!values.TryAdd(item.Name, item.CppValue))
            {
                throw new ArgumentException($"Duplicate hyperparameter '{item.Name}'.", nameof(parameters));
            }
        }
        _parameters = Array.AsReadOnly(items);
        _values = new ReadOnlyDictionary<string, string>(values);
    }

    public IReadOnlyList<TileGymHyperparameter> Parameters => _parameters;
    public string this[string name] => _values[name];
    public int GetInt32(string name) => int.Parse(this[name], CultureInfo.InvariantCulture);
    public override string ToString() => string.Join(", ", _parameters.Select(static p => $"{p.Name}={p.CppValue}"));
}
