@ECHO ON

SET source=\\130.83.179.201\Elements\BP4D+_v0.2\2D+3D\
FOR /F "TOKENS=*" %%F IN ('DIR /S /B "%source%\F002.zip"') DO "C:\Program Files\7-Zip\7z.exe" x "%%~fF" -o"\\130.83.56.65\StudiShare\sarah\Databases\3)Testing\BD4D+\%~1" *.jpg -r
EXIT