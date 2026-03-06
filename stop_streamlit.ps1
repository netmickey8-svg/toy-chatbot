$targets = Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*streamlit run app.py*" }

if (-not $targets) {
  Write-Output "NO_STREAMLIT_PROCESS"
  exit 0
}

foreach ($t in $targets) {
  try {
    Stop-Process -Id $t.ProcessId -Force -ErrorAction Stop
    Write-Output ("STOPPED_PID={0}" -f $t.ProcessId)
  } catch {
    Write-Output ("FAILED_PID={0}" -f $t.ProcessId)
  }
}
