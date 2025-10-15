#!/usr/bin/env pwsh
# download_checkpoints.ps1
# PowerShell script to download IFRNet checkpoints

Write-Host "🚀 IFRNet Checkpoint Download Script (PowerShell)" -ForegroundColor Green
Write-Host "=" * 60

# Create checkpoints directory
$checkpointsDir = Join-Path $PSScriptRoot "checkpoints"
Write-Host "[INFO] Creating checkpoints directory: $checkpointsDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $checkpointsDir | Out-Null

# Download URL
$url = "https://www.dropbox.com/scl/fo/gvfjc8bq259l4cre2ai0k/AIxkWTcEOcvIIYe7RDlZpag?rlkey=x4lxph520gbt0tjy839gmwoc0&e=1&dl=1"
$zipPath = Join-Path $checkpointsDir "ifrnet_checkpoints.zip"

try {
    # Download the file
    Write-Host "[INFO] Downloading checkpoints from Dropbox..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    
    # Extract the ZIP
    Write-Host "[INFO] Extracting checkpoints..." -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $checkpointsDir -Force
    
    # Clean up ZIP file
    Write-Host "[INFO] Cleaning up temporary ZIP file..." -ForegroundColor Yellow
    Remove-Item $zipPath
    
    # List contents
    Write-Host "`n✅ Download complete!" -ForegroundColor Green
    Write-Host "📁 Checkpoints location: $checkpointsDir" -ForegroundColor Cyan
    Write-Host "`n📋 Downloaded files:" -ForegroundColor Cyan
    Get-ChildItem -Path $checkpointsDir -Recurse | ForEach-Object {
        if ($_.PSIsContainer) {
            Write-Host "📂 $($_.FullName)" -ForegroundColor Blue
        } else {
            $sizeMB = [math]::Round($_.Length / 1MB, 1)
            Write-Host "📄 $($_.FullName) ($sizeMB MB)" -ForegroundColor White
        }
    }
    
    Write-Host "`n🎯 Usage example:" -ForegroundColor Green
    Write-Host "python interpolate_video.py --input video.mp4 --target_fps 24 --model ./checkpoints/IFRNet/IFRNet_Vimeo90K.pth" -ForegroundColor Gray
    
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}