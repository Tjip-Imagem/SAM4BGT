@echo off
setlocal enabledelayedexpansion

set PYTHONPATH="C:\Users\Tjip.vanDale\segment-anything-2"
echo %PYTHONPATH%

set "filename=C:\Users\Tjip.vanDale\segment-anything-2\arg_coords.py"

if exist "%filename%" (
    echo Reading file: %filename%
    for /f "delims=" %%i in (%filename%) do (
        set "line=%%i"
        echo !line!
		goto :end
    )
) else (
    echo File not found: %filename%
)


:end


rem set "string=%line%"
rem echo Original string: %string%

:: Extract substring starting at position 5 with length 3
rem set "substring=%string:~,3%"
rem echo Substring: %substring%

rem set /a number=%substring%
rem set "denominator=3"
rem set /a num_iterations=%number% / %denominator%
set "num_iterations=5"
echo Aantal iteraties: %num_iterations%
for /l %%i in (1, 1, %num_iterations%) do (
    python C:/Users/Tjip.vanDale/segment-anything-2/sam_new.py
)

echo Einde batch 
endlocal