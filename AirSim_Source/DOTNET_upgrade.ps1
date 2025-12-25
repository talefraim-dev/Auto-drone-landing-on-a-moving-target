# סקריפט לעדכון גורף של פרויקטים מ-.NET 6.0 ל-.NET 8.0
$RootPath = Get-Location
Write-Host "Starting recursive update in: $RootPath" -ForegroundColor Cyan

# חיפוש כל קבצי ה-csproj בתתי התיקיות
$files = Get-ChildItem -Recurse -Filter *.csproj

foreach ($file in $files) {
    $content = Get-Content $file.FullName
    if ($content -match 'net6.0') {
        Write-Host "Updating: $($file.FullName)" -ForegroundColor Yellow
        $newContent = $content -replace '<TargetFramework>net6.0</TargetFramework>', '<TargetFramework>net8.0</TargetFramework>'
        $newContent = $newContent -replace '<TargetFrameworks>net6.0</TargetFrameworks>', '<TargetFrameworks>net8.0</TargetFrameworks>'
        # החלפת מופעים כלליים של net6.0 (למשל ב-Conditions)
        $newContent = $newContent -replace 'net6.0', 'net8.0'
        
        Set-Content -Path $file.FullName -Value $newContent
    }
}

Write-Host "Done! All projects have been updated to .NET 8.0." -ForegroundColor Green