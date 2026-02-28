// Type 'Program' can be sealed because it has no subtypes in its containing assembly and is not externally visible
#pragma warning disable CA1852
using System;
using System.Diagnostics;
using CudaSharp;
using static System.Console;
[assembly: System.Runtime.InteropServices.ComVisible(false)]

OutputEncoding = System.Text.Encoding.UTF8;
Action<string> log = t => { WriteLine(t); Trace.WriteLine(t); };

log(nameof(nvcuda));

// Above example code is for demonstration purposes only.
// Short names and repeated constants are only for demonstration.
