from pattern import Checker
objChecker=Checker(100,10)
objChecker.show()
from pattern import Circle
objCircle=Circle(100,25,(50,50))
objCircle.show()
from pattern import Spectrum
objSpectrum=Spectrum(100)
objSpectrum.show()
from generator import ImageGenerator
objImageGenerator=ImageGenerator('C:\BABU\FAU\FAU coding\exercise_data','C:\BABU\FAU\FAU coding\Labels.json',11,[32,32,3],True,True,True)
objImageGenerator.show()