# Project Rules and Conventions

## Naming Conventions
* **Exact Matching:** All C# API elements (classes, methods, parameters, and constants) MUST match the original C API names and casing exactly.
* **DLL Mapping:** Each C# file representing a DLL must be named exactly after that DLL (e.g., `nvcuda.cs` for `nvcuda.dll`).
* **Class Names:** The static class containing the P/Invoke bindings must match the DLL name exactly (e.g., `public static partial class nvcuda`).

## File Structure and Nesting
* **API Definition:** API methods must be defined in a file named exactly after the DLL (e.g., `nvcuda.cs`).
* **Type Definitions:** All API-specific types (enums, structs, handles) must be **nested** within the API's static class.
* **Type File:** Define these nested types in a separate partial class file named `[DllName].Types.cs` (e.g., `nvcuda.Types.cs`).
* **Test Files:** Test classes must be named `[DllName]Tests.cs` (e.g., `nvcudaTests.cs`) and contain unit tests specific to that API.

## Implementation Details
* **Type Safety:** Use `readonly record struct` wrappers around `IntPtr` for all handles (e.g., `CUcontext`, `CUdeviceptr`).
* **Enums:** Use enums for all return values (`CUresult`) and bitwise flags.
* **Modern Interop:** Use `LibraryImport` for all P/Invoke declarations.
* **Span Usage:** Use `Span<T>` (for output) and `ReadOnlySpan<T>` (for input) instead of arrays for all interop buffers to ensure type safety and avoid pinning overhead where possible.
* **Pointer Types:** Use specific pointer types where applicable. For untyped data, use `IntPtr` or `void*`. For text buffers, use `Span<byte>` or `Span<char>` as appropriate.

## Workflow
* **Code Quality:** Run `dotnet format` before every commit.
* **Atomic Commits:** Commit after each successful change or logical unit of work.

## Code Quality and Formatting
* **Adhere to .editorconfig:** Strictly follow the formatting rules defined in the `.editorconfig` file.
* **Clean Formatting:** Avoid adding unnecessary empty lines and ensure proper indentation.
* **Performance:** Prioritize writing high-performance, low-overhead code, especially for interop calls.

## Documentation
* **XML Documentation:** All public methods and types MUST have XML documentation summaries.
* **External Links:** Include a `<see href="..."/>` link to the official NVIDIA documentation in the XML remarks or summary.


