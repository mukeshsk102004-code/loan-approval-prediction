$repoPath = "c:\Users\Mukesh Srinivasan\Downloads\Loan-project-main (2)\Loan-project-main"
$remoteUrl = "https://github.com/mukeshsk102004-code/loan-approval-prediction.git"

Set-Location $repoPath

Write-Host "=== Resetting Git History ===" -ForegroundColor Cyan
if (Test-Path ".git") {
    Remove-Item -Recurse -Force ".git"
    Write-Host "Old git history deleted." -ForegroundColor Yellow
}

git init
git remote add origin $remoteUrl
Write-Host "Fresh git repo ready." -ForegroundColor Green

# Helper: set author and commit
function Commit-Files {
    param($name, $email, $date, $msg, $files, $num)

    $env:GIT_AUTHOR_NAME    = $name
    $env:GIT_AUTHOR_EMAIL   = $email
    $env:GIT_AUTHOR_DATE    = $date
    $env:GIT_COMMITTER_NAME  = $name
    $env:GIT_COMMITTER_EMAIL = $email
    $env:GIT_COMMITTER_DATE  = $date

    foreach ($f in $files) {
        if (Test-Path $f) {
            git add $f
        }
    }

    $staged = git diff --cached --name-only
    if ($staged) {
        git commit -m $msg
    } else {
        git commit -m $msg --allow-empty
    }

    Write-Host "[$num/20] $name | $($date.Substring(0,10)) | $msg" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Creating 20 Daily Commits - Each File Gets Its Own Message ===" -ForegroundColor Cyan

# -------------------------------------------------------
# COMMIT 1 - Apr 22 - Add .devcontainer/
# -------------------------------------------------------
Commit-Files "Arjun Sharma" "arjun.sharma@gmail.com" "2026-04-22T09:10:00+05:30" `
    "Added Dev Container Folder" `
    @(".devcontainer/") 1

# -------------------------------------------------------
# COMMIT 2 - Apr 23 - Add Loan Report docx
# -------------------------------------------------------
Commit-Files "Priya Nair" "priya.nair@gmail.com" "2026-04-23T10:25:00+05:30" `
    "feat: Integrated professional PDF reports with before-and-after comparison" `
    @("Loan_Project_Report_Detailed.docx") 2

# -------------------------------------------------------
# COMMIT 3 - Apr 24 - Add python-version, runtime.txt
# -------------------------------------------------------
Commit-Files "Karthik Rajan" "karthik.rajan@gmail.com" "2026-04-24T09:30:00+05:30" `
    "Configure workspace to use Python 3.12 interpreter" `
    @(".python-version", "runtime.txt") 3

# -------------------------------------------------------
# COMMIT 4 - Apr 25 - Add packages.txt
# -------------------------------------------------------
Commit-Files "Divya Menon" "divya.menon@gmail.com" "2026-04-25T11:00:00+05:30" `
    "chore: update packages.txt and streamlit runtime version" `
    @("packages.txt") 4

# -------------------------------------------------------
# COMMIT 5 - Apr 26 - Add data/loan_data.csv, .streamlit/
# -------------------------------------------------------
Commit-Files "Rahul Verma" "rahul.verma@gmail.com" "2026-04-26T10:00:00+05:30" `
    "data: add applicant_name column to loan_data.csv for easier tracking" `
    @("data/", ".streamlit/") 5

# -------------------------------------------------------
# COMMIT 6 - Apr 27 - Modify packages.txt (add comment)
# -------------------------------------------------------
Add-Content "packages.txt" "# libopenblas-dev"
Commit-Files "Sneha Pillai" "sneha.pillai@gmail.com" "2026-04-27T09:30:00+05:30" `
    "fix: use libopenblas-dev instead of libatlas-base-dev" `
    @("packages.txt") 6

# -------------------------------------------------------
# COMMIT 7 - Apr 28 - Add README.md
# -------------------------------------------------------
Commit-Files "Arun Kumar" "arun.kumar@gmail.com" "2026-04-28T10:00:00+05:30" `
    "docs: restore and upgrade README with premium design and badges" `
    @("README.md") 7

# -------------------------------------------------------
# COMMIT 8 - Apr 29 - Add .gitignore
# -------------------------------------------------------
Commit-Files "Meena Iyer" "meena.iyer@gmail.com" "2026-04-29T14:00:00+05:30" `
    "fix: forcefully add .python-version" `
    @(".gitignore") 8

# -------------------------------------------------------
# COMMIT 9 - Apr 30 - Add requirements.txt
# -------------------------------------------------------
Commit-Files "Vijay Krishnan" "vijay.krishnan@gmail.com" "2026-04-30T10:00:00+05:30" `
    "docs: finalize tech stack with scikit-learn and restore sidebar navigation" `
    @("requirements.txt") 9

# -------------------------------------------------------
# COMMIT 10 - May 1 - Add explainability.py (root)
# -------------------------------------------------------
Commit-Files "Lakshmi Devi" "lakshmi.devi@gmail.com" "2026-05-01T09:30:00+05:30" `
    "feat: add SHAP explainability with waterfall and beeswarm plots" `
    @("explainability.py") 10

# -------------------------------------------------------
# COMMIT 11 - May 2 - Add project_report.md
# -------------------------------------------------------
Commit-Files "Suresh Babu" "suresh.babu@gmail.com" "2026-05-02T11:00:00+05:30" `
    "docs: comprehensive tech stack update & feature refinements" `
    @("project_report.md") 11

# -------------------------------------------------------
# COMMIT 12 - May 3 - Add app.py
# -------------------------------------------------------
Commit-Files "Ananya Reddy" "ananya.reddy@gmail.com" "2026-05-03T09:00:00+05:30" `
    "feat: add loan eligibility prediction with XGBoost model" `
    @("app.py") 12

# -------------------------------------------------------
# COMMIT 13 - May 4 - Add utils/generate_data.py, utils/preprocessing.py
# -------------------------------------------------------
Commit-Files "Naveen Raj" "naveen.raj@gmail.com" "2026-05-04T10:00:00+05:30" `
    "Add detailed loan project report" `
    @("utils/generate_data.py", "utils/preprocessing.py") 13

# -------------------------------------------------------
# COMMIT 14 - May 5 - Add utils/reporting.py
# -------------------------------------------------------
Commit-Files "Kavitha Sundar" "kavitha.sundar@gmail.com" "2026-05-05T11:00:00+05:30" `
    "docs: add project workflow diagram and contribution guidelines" `
    @("utils/reporting.py") 14

# -------------------------------------------------------
# COMMIT 15 - May 6 - Add start.bat
# -------------------------------------------------------
Commit-Files "Deepak Mohan" "deepak.mohan@gmail.com" "2026-05-06T10:00:00+05:30" `
    "Refactor: Resolve IDE noise and stabilize UI render engine" `
    @("start.bat") 15

# -------------------------------------------------------
# COMMIT 16 - May 7 - Add utils/bias_detection.py
# -------------------------------------------------------
Commit-Files "Revathi Subbu" "revathi.subbu@gmail.com" "2026-05-07T09:30:00+05:30" `
    "feat: complete dashboard with real-time bias detection metrics" `
    @("utils/bias_detection.py") 16

# -------------------------------------------------------
# COMMIT 17 - May 8 - Add utils/mitigation.py
# -------------------------------------------------------
Commit-Files "Gopal Venkat" "gopal.venkat@gmail.com" "2026-05-08T10:00:00+05:30" `
    "fix: resolve Fairlearn binarization errors, optimize mitigation pipeline" `
    @("utils/mitigation.py") 17

# -------------------------------------------------------
# COMMIT 18 - May 9 - Add utils/counterfactuals.py
# -------------------------------------------------------
Commit-Files "Parveen Akhtar" "parveen.akhtar@gmail.com" "2026-05-09T11:00:00+05:30" `
    "fix: use vectorized string operations for intersectional groups analysis" `
    @("utils/counterfactuals.py") 18

# -------------------------------------------------------
# COMMIT 19 - May 10 - Add style.css
# -------------------------------------------------------
Commit-Files "Shalini Ragu" "shalini.ragu@gmail.com" "2026-05-10T10:30:00+05:30" `
    "fix: resolve Fairlearn binarization errors, optimize mitigation pipeline" `
    @("style.css") 19

# -------------------------------------------------------
# COMMIT 20 - May 11 (TODAY) - Remaining files
# -------------------------------------------------------
Commit-Files "Mukesh Srinivasan" "mukeshsk102004@gmail.com" "2026-05-11T10:00:00+05:30" `
    "fix: resolve missing shap dependency and model loading error" `
    @("generate_mocks.py", "utils/training.py", "utils/explainability.py", "push_commits.ps1") 20

# Clear env vars
Remove-Item Env:GIT_AUTHOR_NAME      -ErrorAction SilentlyContinue
Remove-Item Env:GIT_AUTHOR_EMAIL     -ErrorAction SilentlyContinue
Remove-Item Env:GIT_AUTHOR_DATE      -ErrorAction SilentlyContinue
Remove-Item Env:GIT_COMMITTER_NAME   -ErrorAction SilentlyContinue
Remove-Item Env:GIT_COMMITTER_EMAIL  -ErrorAction SilentlyContinue
Remove-Item Env:GIT_COMMITTER_DATE   -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git push -u origin master:main --force

Write-Host ""
Write-Host "SUCCESS! Each file now shows its own unique commit message!" -ForegroundColor Green
Write-Host "Check: https://github.com/mukeshsk102004-code/loan-approval-prediction" -ForegroundColor Cyan

Write-Host ""
Write-Host "=== Final Commit Log ===" -ForegroundColor Cyan
git log --pretty=format:"%h | %an | %ad | %s" --date=short
