# Locate or install Python for the Windows launcher, then run the modeling tool.
$ErrorActionPreference = 'Stop'
$toolDir = Join-Path $env:LOCALAPPDATA 'DSpaLaRefiner'
$runtime = Join-Path $toolDir 'Python313'
$managedPython = Join-Path $runtime 'python.exe'
$probe = 'import struct,sys,venv; sys.exit(0 if sys.version_info >= (3,10) and struct.calcsize(chr(80)) == 8 else 1)'
$python = $null
$pythonArgs = @()

try {
    Write-Host 'Starting DSpaLaRefiner. Checking for 64-bit Python...'
    if (Test-Path -LiteralPath $managedPython) {
        $python = $managedPython
    } else {
        foreach ($name in @('py.exe', 'python.exe')) {
            $candidate = Get-Command $name -ErrorAction SilentlyContinue
            if (-not $candidate -or $candidate.Source -match '\\Microsoft\\WindowsApps\\') { continue }
            $prefix = @()
            if ($name -eq 'py.exe') { $prefix = @('-3') }
            & $candidate.Source @prefix -c $probe
            if ($LASTEXITCODE -eq 0) {
                $python = $candidate.Source
                $pythonArgs = $prefix
                break
            }
        }
    }

    if (-not $python) {
        if (-not [Environment]::Is64BitOperatingSystem) { throw 'This launcher requires 64-bit Windows.' }
        $bootstrap = Join-Path $toolDir 'bootstrap'
        New-Item -ItemType Directory -Force -Path $bootstrap | Out-Null
        $installer = Join-Path $bootstrap 'python-3.13.17-amd64.exe'
        Write-Host 'Python was not found. Downloading the official Python installer...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        if (-not (Test-Path -LiteralPath $installer)) {
            $partial = "$installer.part"
            $url = 'https://www.python.org/ftp/python/3.13.17/python-3.13.17-amd64.exe'
            try {
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 180 -Uri $url -OutFile $partial
            } catch {
                Write-Host 'Download was interrupted. Retrying in smaller blocks...'
                $length = 29941280
                $step = 1MB
                $output = [IO.File]::Create($partial)
                try {
                    for ($start = 0; $start -lt $length; $start += $step) {
                        $end = [Math]::Min($start + $step, $length) - 1
                        for ($attempt = 1; ; $attempt++) {
                            try {
                                $request = [Net.HttpWebRequest]::Create($url)
                                $request.Timeout = 30000
                                $request.ReadWriteTimeout = 30000
                                $request.AddRange($start, $end)
                                $response = $request.GetResponse()
                                $buffer = New-Object IO.MemoryStream
                                try {
                                    if ($response.Headers['Content-Range'] -ne "bytes $start-$end/$length") { throw 'Unexpected Python download range.' }
                                    $response.GetResponseStream().CopyTo($buffer)
                                    if ($buffer.Length -ne ($end - $start + 1)) { throw 'Incomplete Python download block.' }
                                    $buffer.WriteTo($output)
                                } finally {
                                    $buffer.Dispose()
                                    $response.Dispose()
                                }
                                break
                            } catch {
                                if ($attempt -ge 3) { throw }
                            }
                        }
                        Write-Progress -Activity 'Downloading Python' -PercentComplete (100 * ($end + 1) / $length)
                    }
                } finally {
                    $output.Dispose()
                    Write-Progress -Activity 'Downloading Python' -Completed
                }
            }
            Move-Item -LiteralPath $partial -Destination $installer -Force
        }
        $expectedHash = "5da012101ba2e17113a9a1cd901e9ce1441086f2b7ecb5de06ec94163899a190"
        $sha256 = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($installer)
        try {
            $actualHash = [BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '')
        } finally {
            $stream.Dispose()
            $sha256.Dispose()
        }
        if ($actualHash -ne $expectedHash) {
            Remove-Item -LiteralPath $installer
            throw 'The Python installer did not match the published checksum. Start again to download a new copy.'
        }
        Write-Host 'Installing Python for your Windows account...'
        $log = Join-Path $bootstrap 'python-install.log'
        $options = "/quiet /log `"$log`" InstallAllUsers=0 TargetDir=`"$runtime`" Include_launcher=0 Include_test=0 Include_doc=0 Include_tcltk=0 Include_pip=1 PrependPath=0 AssociateFiles=0 Shortcuts=0"
        $setup = Start-Process -FilePath $installer -ArgumentList $options -WindowStyle Hidden -Wait -PassThru
        if ($setup.ExitCode -ne 0 -and $setup.ExitCode -ne 3010) {
            throw "Python installer exited with code $($setup.ExitCode). Log: $log"
        }
        if (-not (Test-Path -LiteralPath $managedPython)) { throw "Python installation did not create $managedPython." }
        $python = $managedPython
    }

    Write-Host 'Python is ready. Starting the modeling tool...'
    & $python @pythonArgs -u (Join-Path $PSScriptRoot 'launch.py') @args
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

