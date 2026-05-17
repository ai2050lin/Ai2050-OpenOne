$fs = [System.IO.FileStream]::new('d:\Ai2050\TransformerLens-Project\tests\glm5_temp\phase199_glm4_log.txt', [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
$sr = [System.IO.StreamReader]::new($fs)
$content = $sr.ReadToEnd()
$sr.Close()
$fs.Close()
$lines = $content -split "`n"
$total = $lines.Count
Write-Output "Total lines: $total"
$start = [Math]::Max(0, $total - 25)
for($i=$start; $i -lt $total; $i++){Write-Output $lines[$i]}
