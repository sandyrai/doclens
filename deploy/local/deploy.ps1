# PowerShell wrapper: runs the bash deploy script with Git Bash.
#   .\deploy.ps1 ats
#   .\deploy.ps1 all
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }
& $bash "$PSScriptRoot/deploy" @args
exit $LASTEXITCODE
