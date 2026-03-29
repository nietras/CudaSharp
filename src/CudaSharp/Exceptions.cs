namespace CudaSharp;

public class CudaException(string message) : Exception(message);

public sealed class CudaException<TResult>(TResult result, string message) : Exception(message)
    where TResult : unmanaged, Enum
{
    public TResult Result { get; } = result;
}
