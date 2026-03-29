using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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

        public static void EnumValuesOkThrows<TEnum>(Func<TEnum, bool> isOk, Action<TEnum> ok)
            where TEnum : unmanaged, Enum
        {
            var values = Enum.GetValues<TEnum>();
            foreach (var value in values)
            {
                if (isOk(value)) { ok(value); }
                else
                {
                    var e = Assert.Throws<CudaException<TEnum>>(() => ok(value));
                    Assert.AreEqual(value.ToString(), e.Message);
                }
            }
        }
    }
}

