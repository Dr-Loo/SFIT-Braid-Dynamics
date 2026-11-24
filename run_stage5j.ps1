# run_stage5j.ps1 -- Stage 5J frozen plan. Run only AFTER stage5j_predictions.json is timestamped.
# Usage:  .\run_stage5j.ps1 -ExpectedSha <hash printed by stage5j_build_predictions.py and timestamped>
param([Parameter(Mandatory=$true)][string]$ExpectedSha)
$ErrorActionPreference = "Stop"
$plan = Get-Content stage5j_predictions.json -Raw | ConvertFrom-Json
foreach ($f in "stage5j_primary_output.jsonl", "stage5j_plainloop_output.jsonl") {
  if (Test-Path $f) { throw "$f already exists: the protocol forbids re-running. Move it aside and record why." }
}
foreach ($p in $plan.run_plan.primary) {
  Write-Host "primary r=$($p.r)"
  $line = python stage5j_primary_sim.py $p.r $p.G $p.C $p.seed
  if ($LASTEXITCODE -ne 0) { throw "primary r=$($p.r) exited with $LASTEXITCODE" }
  Add-Content -Path stage5j_primary_output.jsonl -Value $line -Encoding UTF8
}
foreach ($p in $plan.run_plan.plainloop) {
  Write-Host "plain-loop r=$($p.r)"
  $line = python stage5j_plainloop_sim.py $p.r $p.G $p.T $p.seed
  if ($LASTEXITCODE -ne 0) { throw "plain-loop r=$($p.r) exited with $LASTEXITCODE" }
  Add-Content -Path stage5j_plainloop_output.jsonl -Value $line -Encoding UTF8
}
python stage5j_score.py stage5j_predictions.json stage5j_primary_output.jsonl stage5j_plainloop_output.jsonl $ExpectedSha
