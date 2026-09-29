namespace CudaSharp.Tester;

static class TileGymValidation
{
    public static void Validate(this CudaBuffer<float> actual, ReadOnlySpan<float> expected,
        string kernel, float absoluteTolerance = 2e-5f, float relativeTolerance = 2e-5f)
    {
        var host = actual.CopyToHost();
        Validate(host, expected, kernel, absoluteTolerance, relativeTolerance);
    }

    public static void Validate(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected,
        string kernel, float absoluteTolerance = 2e-5f, float relativeTolerance = 2e-5f)
    {
        if (actual.Length != expected.Length)
        {
            throw new InvalidOperationException($"{kernel} validation length mismatch.");
        }
        for (var i = 0; i < actual.Length; i++)
        {
            var tolerance = absoluteTolerance + relativeTolerance * Math.Abs(expected[i]);
            if (!float.IsFinite(actual[i]) || Math.Abs(actual[i] - expected[i]) > tolerance)
            {
                throw new InvalidOperationException(
                    $"{kernel} validation failed at {i}: {actual[i]} != {expected[i]} (tolerance {tolerance}).");
            }
        }
    }
}
