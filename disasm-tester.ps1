param(
    [string]$runtime = "win-arm64"
)
dotnet publish src/CudaSharp.Tester/CudaSharp.Tester.csproj -c Release -r "$runtime" -f net10.0 --self-contained true /p:PublishAot=true /p:DebugSymbols=true
dumpbin /DISASM /SYMBOLS "artifacts\publish\CudaSharp.Tester\release_$runtime\CudaSharp.Tester.exe" > "artifacts\publish\CudaSharp.Tester\release_$runtime\disassembly.asm"
