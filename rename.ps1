#!/usr/bin/env pwsh
$oldName = "CudaSharp"
$newName = "CudaSharp"
Get-ChildItem -Filter "*$oldName*" -Recurse | Rename-Item -NewName {$_.name -replace $oldName, $newName }
Get-ChildItem -Recurse -Include "*.sln","*.slnx","*.cs","*.xaml","*.xml","*.csproj","*.xproj","*.json","*.md","*.cmd","*.props","*.txt","*.yml","*.ps1" |
 ForEach-Object { $a = $_.fullname; ( [System.IO.File]::ReadAllText($a) ) | ForEach-Object {
	 If ($_.Contains($oldName))
	 {
		 $newContent = $_.Replace($oldName, $newName)
		 #$newContent
		 [System.IO.File]::WriteAllText($a, $newContent, [System.Text.Encoding]::UTF8)
		 "Changed: " + $a
	 }
	}
}