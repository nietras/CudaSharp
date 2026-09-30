using System.Collections.Generic;

namespace CudaSharp.Tester;

/// <summary>Measures one compilation attempt and its shared module load, without repeating costs per specialization.</summary>
sealed record TileGymBatchTiming(int Id, string Header, IReadOnlyList<TileGymKernelSpec> Specializations,
    double CompileMilliseconds, double LoadMilliseconds = 0, Exception? Error = null,
    long CompileStartTimestamp = 0, long LoadStartTimestamp = 0, int TileIrBytes = 0);
