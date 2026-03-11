@echo off
setlocal EnableDelayedExpansion

:: ========================================================
::                CONFIGURATION (SHARED)
:: ========================================================
cd /d "%~dp0"

:: Thresholds
set MAX_CPU=90
set MAX_GPU=90
set WAIT_SECONDS=10

:: Paths
set "FFMPEG_CMD=C:\Users\gl_pc\Desktop\code\ffmpeg-2026-03-09-git-9b7439c31b-full_build\bin\ffmpeg.exe"
set "ONNX_WEIGHTS_PATH=C:\Users\gl_pc\Desktop\data\yolov/bests_1280_mosiac_close.pt"
set "TRODES_EXPORT_CMD=C:\Users\gl_pc\Desktop\Trodes_2-8-0_Windows11\trodesexport.exe"
set FREQ=30000

:: ========================================================
::            MODE CHECK: MASTER OR WORKER?
:: ========================================================
:: If a 4th argument exists, it's the user's selection passed from Master
if "%~1"==":WORKER" (
    set "STEPS_TO_RUN=%~4"
    goto :WORKER_ROUTINE
)

echo ========================================================
echo           SMART PARALLEL MODE (Multi-Step)
echo ========================================================
echo [CONFIG] Max CPU: %MAX_CPU%%% ^| Max GPU: %MAX_GPU%%%
echo.

:: 3. Handle Input (Master Mode)
if "%~1"=="" (
    echo Usage: runner_windows.bat "path_to_data_folder"
    exit /b 1
)

:: --- NEW: STEP SELECTION MENU ---
echo Select steps to run (e.g., 123 for steps 1, 2, and 3):
echo [1] Trodes Export
echo [2] Sync Script
echo [3] Stitching
echo [4] Tracker
echo [5] Plotting
echo [6] Compression
echo [7] Sorting
echo [8] LFP
echo [d] LFP
echo [9] Cleaning
echo.
set /p "MY_SELECTION=Enter steps: "

pushd "%~1"
set "ROOT_DIR=%CD%"
popd
echo [DEBUG] Target Root Directory: [%ROOT_DIR%]

:: 4. Scan Loop (Master Mode)
set count=0

for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    set "NUM=!DIR_NAME:ip=!"
    set "OP_PATH=%ROOT_DIR%\op!NUM!"

    if exist "!OP_PATH!\" (
        echo.
        echo [QUEUE] Preparing: !DIR_NAME!
        
        call :WAIT_FOR_RESOURCES
        
        set /a count+=1
        :: Pass the %MY_SELECTION% as the 4th parameter to the worker
        start "Job-!DIR_NAME!" cmd /k call "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!" "%MY_SELECTION%"
        
        timeout /t 3 /nobreak >nul
    )
)

echo.
echo ========================================================
echo [MASTER] Launched !count! jobs with steps: %MY_SELECTION%
echo ========================================================
pause
exit /b

:: ========================================================
::             RESOURCE MONITOR SUBROUTINE
:: ========================================================
:WAIT_FOR_RESOURCES
:CHECK_AGAIN
    set CPU_LOAD=0
    for /f "skip=1" %%P in ('wmic cpu get loadpercentage') do (
        if "%%P" neq "" set CPU_LOAD=%%P
        goto :break_cpu
    )
    :break_cpu

    set GPU_LOAD=0
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits > gpu_temp.txt 2>nul
    if exist gpu_temp.txt (
        set /p GPU_LOAD=<gpu_temp.txt
        del gpu_temp.txt
    )
    if "%GPU_LOAD%"=="" set GPU_LOAD=0

    if !CPU_LOAD! GTR %MAX_CPU% (
        echo     [WAIT] High CPU: !CPU_LOAD!%%. Pausing...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )
    if !GPU_LOAD! GTR %MAX_GPU% (
        echo     [WAIT] High GPU: !GPU_LOAD!%%. Pausing...
        timeout /t %WAIT_SECONDS% /nobreak >nul
        goto :CHECK_AGAIN
    )
    echo     [CHECK] CPU: !CPU_LOAD!%% ^| GPU: !GPU_LOAD!%% - OK.
exit /b

:: ========================================================
::                THE WORKER SUBROUTINE
:: ========================================================
:WORKER_ROUTINE
set "IP=%~2"
set "OP=%~3"
color 0A 

echo.
echo [INFO] Running steps [%STEPS_TO_RUN%] for !IP!

:: --- STEP 1 ---
echo %STEPS_TO_RUN% | findstr "1" >nul
if %errorlevel% equ 0 (
    echo [STEP 1] Running Trodes...
    if exist "%TRODES_EXPORT_CMD%" (
        for %%F in ("%IP%\*.rec") do ("%TRODES_EXPORT_CMD%" -dio -raw -rec "%%F")
    )
)

:: --- STEP 2 ---
echo %STEPS_TO_RUN% | findstr "2" >nul
if %errorlevel% equ 0 (
    echo [STEP 2] Running Sync Script...
    if exist ".\src\Video_LED_Sync_using_ICA.py" (
        python -u ./src/Video_LED_Sync_using_ICA.py -i "%IP%" -o "%OP%" -f %FREQ%
    )
)

:: --- STEP 3 ---
echo %STEPS_TO_RUN% | findstr "3" >nul
if %errorlevel% equ 0 (
    echo [STEP 3] Running Stitching...
    if exist ".\src\join_views.py" (
        python -u ./src/join_views.py "%IP%"
    )
)

:: --- STEP 4 ---
echo %STEPS_TO_RUN% | findstr "4" >nul
if %errorlevel% equ 0 (
    echo [STEP 4] Running Tracker...
    if exist "%IP%\stitched.mp4" (
        python -u ./src/TrackerYolov11.py --input_folder "%IP%" --output_folder "%OP%" --onnx_weight "%ONNX_WEIGHTS_PATH%"
    )
)

:: --- STEP 5 ---
echo %STEPS_TO_RUN% | findstr "5" >nul
if %errorlevel% equ 0 (
    echo [STEP 5] Running Plotting...
    if exist ".\src\plot_trials.py" (
        python -u ./src/plot_trials.py -o "%OP%"
    )
)

:: --- STEP 6 ---
echo %STEPS_TO_RUN% | findstr "6" >nul
if %errorlevel% equ 0 (
    echo [STEP 6] Running Compression...
    set "VIDEO_FILE="
    for %%f in ("%OP%\*.mp4") do (set "VIDEO_FILE=%%~f" & goto :FOUND_VIDEO)
    :FOUND_VIDEO
    if not "!VIDEO_FILE!"=="" (
        set "TEMP_FILE=%OP%\__temp_compressed.mp4"
        
        :: CHANGED LINE BELOW to use %FFMPEG_CMD%
        "%FFMPEG_CMD%" -y -v error -i "!VIDEO_FILE!" -vcodec libx264 -crf 28 "!TEMP_FILE!"
        
        if exist "!TEMP_FILE!" move /Y "!TEMP_FILE!" "!VIDEO_FILE!" >nul
        echo [SUCCESS] Video compressed.
    )
)

:: --- STEP 7 ---
echo %STEPS_TO_RUN% | findstr "7" >nul
if %errorlevel% equ 0 (
    echo [STEP 7] Running Sorting...
    if exist ".\src\sorter\sorting.py" (
        python -u ./src/sorter/sorting.py --input_folder "%IP%" --output_folder "%OP%"
    )

)

:: --- STEP 8 ---
echo %STEPS_TO_RUN% | findstr "8" >nul
if %errorlevel% equ 0 (
    echo [STEP 8] Running LFP Extraction...
    if exist ".\src\sorter\export_lfp.py" (
        python -u ./src/sorter/export_lfp.py --input_folder "%IP%" --output_folder "%OP%"
    )

)



python -u ./src/sorter/sorting.py --input_folder "C:\Users\gl_pc\Desktop\data\yolov\error\ip4" --output_folder "C:\Users\gl_pc\Desktop\data\yolov\error\op4"



echo.
echo [COMPLETE] Worker finished.
timeout /t 15
exit