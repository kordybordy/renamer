# ------------------------------------------------------------
# Renamer test environment generator
# - Creates 60 case folders (30 SURNAME_NAME, 30 NAME_SURNAME; ASCII only)
# - Copies one PDF 60 times named "Surname Name - Pozew.pdf" (Polish letters)
# Paths:
#   Source PDFs: C:\renamer\test\skany\
#   Output PDFs: C:\renamer\test\input\
#   Case folders: C:\renamer\test\cases\
# ------------------------------------------------------------

$SkanyDir = "C:\renamer\test\skany"
$InputDir = "C:\renamer\test\input"
$CasesDir = "C:\renamer\test\cases"

New-Item -ItemType Directory -Force -Path $SkanyDir, $InputDir, $CasesDir | Out-Null

# Pick a source PDF (first one found)
$SourcePdf = Get-ChildItem -Path $SkanyDir -Filter *.pdf -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $SourcePdf) {
  Write-Error "No PDF found in $SkanyDir. Put at least one .pdf there and rerun."
  exit 1
}

# --- Pools ---
# ASCII-only for folder names (no Polish letters)
$SurnamesASCII = @(
  "KOWALSKI","NOWAK","WISNIEWSKI","WOJCIK","KOWALCZYK","KAMINSKI","LEWANDOWSKI","ZIELINSKI","SZYMANDOWSKI","DABROWSKI",
  "PIOTROWSKI","MAZUR","KROL","KACZMAREK","JANKOWSKI","WRZESINSKI","PAWLAK","MICHALSKI","NOWICKI","ADAMCZYK",
  "DUDZIAK","GORSKI","CHMIELEWSKI","KOZLOWSKI","KONIECZNY","BARTOSZ","WALCZAK","OLSZEWSKI","WROBEL","JABLONSKI",
  "KRUPA","SZCZEPANSKI","URBAN","WIECZOREK","TOMASZEWSKI","MAJEWSKI","SADOWSKI","ZAKRZEWSKI","JASINSKI","DUDA"
)
$NamesASCII = @(
  "Jan","Anna","Piotr","Katarzyna","Pawel","Agnieszka","Tomasz","Magdalena","Marek","Joanna",
  "Krzysztof","Barbara","Andrzej","Monika","Michal","Ewa","Marcin","Zofia","Adam","Aleksandra",
  "Karol","Natalia","Jakub","Iwona","Rafal","Julia","Dariusz","Oliwia","Patryk","Weronika",
  "Damian","Elzbieta","Grzegorz","Marta","Mateusz","Dorota","Sebastian","Kinga","Lukasz","Beata"
)

# Polish-letter pool for PDF filenames
$SurnamesPL = @(
  "Kowalski","Nowak","Wiśniewski","Wójcik","Kowalczyk","Kamiński","Lewandowski","Zieliński","Szymański","Dąbrowski",
  "Piotrowski","Mazur","Król","Kaczmarek","Jankowski","Wrzesiński","Pawlak","Michałski","Nowicki","Adamczyk",
  "Dudziak","Górski","Chmielewski","Kozłowski","Konieczny","Bartosz","Walczak","Olszewski","Wróbel","Jabłoński",
  "Krupa","Szczepański","Urban","Wieczorek","Tomaszewski","Majewski","Sadowski","Zakrzewski","Jasiński","Duda"
)
$NamesPL = @(
  "Jan","Anna","Piotr","Katarzyna","Paweł","Agnieszka","Tomasz","Magdalena","Marek","Joanna",
  "Krzysztof","Barbara","Andrzej","Monika","Michał","Ewa","Marcin","Zofia","Adam","Aleksandra",
  "Karol","Natalia","Jakub","Iwona","Rafał","Julia","Dariusz","Oliwia","Patryk","Weronika",
  "Damian","Elżbieta","Grzegorz","Marta","Mateusz","Dorota","Sebastian","Kinga","Łukasz","Beata"
)

function Get-Rand($arr) { $arr | Get-Random }

# Helper: generate folder-safe name part (already ASCII for folder pools)
function To-FolderToken([string]$s) {
  # Ensure only A-Z0-9_ (keep it simple and Windows-safe)
  $t = $s.ToUpper()
  $t = $t -replace "[^A-Z0-9]", ""
  return $t
}

# Create 30 folders: SURNAME_NAME (some multi-defendant: SURNAME_SURNAME)
$folders = New-Object System.Collections.Generic.List[string]

for ($i=0; $i -lt 30; $i++) {
  $s1 = To-FolderToken (Get-Rand $SurnamesASCII)

  # ~35% chance of multiple defendants in folder name
  $multi = ((Get-Random -Minimum 1 -Maximum 101) -le 35)

  if ($multi) {
    $s2 = To-FolderToken (Get-Rand $SurnamesASCII)
    while ($s2 -eq $s1) { $s2 = To-FolderToken (Get-Rand $SurnamesASCII) }
    $folderName = "${s1}_${s2}"
  } else {
    $n1 = To-FolderToken (Get-Rand $NamesASCII)
    $folderName = "${s1}_${n1}"
  }

  $full = Join-Path $CasesDir $folderName
  New-Item -ItemType Directory -Force -Path $full | Out-Null
  $folders.Add($folderName) | Out-Null
}

# Create 30 folders: NAME_SURNAME
for ($i=0; $i -lt 30; $i++) {
  $n = To-FolderToken (Get-Rand $NamesASCII)
  $s = To-FolderToken (Get-Rand $SurnamesASCII)
  $folderName = "${n}_${s}"

  $full = Join-Path $CasesDir $folderName
  New-Item -ItemType Directory -Force -Path $full | Out-Null
  $folders.Add($folderName) | Out-Null
}

# Create 60 PDFs in input folder named with Polish letters: "Surname Name - Pozew.pdf"
# (Use unique pairs; if collision, append (n))
$used = @{}
for ($i=0; $i -lt 60; $i++) {
  $surname = Get-Rand $SurnamesPL
  $name    = Get-Rand $NamesPL
  $base    = "$surname $name - Pozew"
  $file    = "$base.pdf"

  if ($used.ContainsKey($file)) {
    $used[$file]++
    $file = "$base ($($used[$file])).pdf"
  } else {
    $used[$file] = 0
  }

  Copy-Item -LiteralPath $SourcePdf.FullName -Destination (Join-Path $InputDir $file) -Force
}

Write-Host "DONE."
Write-Host "Source PDF: $($SourcePdf.FullName)"
Write-Host "Case folders created: 60 in $CasesDir"
Write-Host "Test PDFs created: 60 in $InputDir"
