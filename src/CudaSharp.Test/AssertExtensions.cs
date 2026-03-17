using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static CudaSharp.nvcuda;

namespace CudaSharp.Test;

public static class AssertExtensions
{
    extension(Assert)
    {
        public static void EnumValuesToString<TEnum>(Func<TEnum, string> toString)
            where TEnum : unmanaged, Enum
        {
            var values = Enum.GetValues<TEnum>();
            foreach (var value in values)
            {
                Assert.AreEqual(value.ToString(), toString(value));
            }
        }
    }
}

