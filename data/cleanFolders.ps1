# Set the root folder you want to clean (change this or run from the desired directory)
$rootFolder = "."  # <-- Change this to your target folder, or use "." for current

# If you want to run it in the current directory, uncomment the next line instead:
# $rootFolder = Get-Location

Get-ChildItem -Path $rootFolder -Recurse -Directory | 
Sort-Object -Property FullName -Descending |  # Important: bottom-up to avoid parent-before-child issues
Where-Object {
    # Check if the folder is empty (no files AND no subfolders)
    (Get-ChildItem -Path $_.FullName -Force | Measure-Object).Count -eq 0
} |
ForEach-Object {
    Write-Host "Removing empty folder: $($_.FullName)"
    Remove-Item -Path $_.FullName -Force
}

Write-Host "Cleanup complete!" -ForegroundColor Green