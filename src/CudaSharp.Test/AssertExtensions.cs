using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CudaSharp.Test;

public static class AssertExtensions
{
    extension(Assert)
    {
        [return: NotNullIfNotNull(nameof(actual))]
        public static T? AreEqualReturn<T>(T? expected, T? actual, string? message = "",
            [CallerArgumentExpression(nameof(expected))] string expectedExpression = "",
            [CallerArgumentExpression(nameof(actual))] string actualExpression = "")
        {
            Assert.AreEqual(expected, actual, message, expectedExpression, actualExpression);
            return actual;
        }

        public static void AreSame<T>(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, string? message = "",
            [CallerArgumentExpression(nameof(expected))] string expectedExpression = "",
            [CallerArgumentExpression(nameof(actual))] string actualExpression = "")
        {
            Assert.AreEqual(expected.Length, actual.Length, message, expectedExpression, actualExpression);
            Assert.IsTrue(Unsafe.AreSame(ref MemoryMarshal.GetReference(expected),
                                         ref MemoryMarshal.GetReference(actual)),
                          message);
        }
    }
}
